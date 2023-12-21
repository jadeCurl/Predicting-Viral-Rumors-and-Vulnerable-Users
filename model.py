import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import argparse
import numpy as np
from scipy.linalg import block_diag
from sklearn.metrics import  accuracy_score, precision_score, recall_score, f1_score
from torchmeta.modules import MetaModule, MetaSequential, MetaLinear, MetaBatchNorm1d
from torch.utils.data import DataLoader
from dgl_format import *

from collections import OrderedDict


def update_parameters_gd(model, loss, step_size=0.5, first_order=False, log_var=None):
    """Update the parameters of the model, with one step of gradient descent."""
    # print(loss)
    grads = torch.autograd.grad(loss, model.meta_parameters(),
        create_graph=not first_order, allow_unused=True) # allow_unused is necessary for when not all the output heads are used

    params = OrderedDict()
    for (name, param), grad in zip(model.meta_named_parameters(), grads):
        if grad is None: # the gradients of the output heads that are not used will be None
            continue
        params[name] = param - step_size * grad
    return params

def batch2tensor(batch_adj, batch_feat, node_per_pool_graph):
	"""
	transform a batched graph to batched adjacency tensor and node feature tensor
	"""
	batch_size = int(batch_adj.size()[0] / node_per_pool_graph)
	adj_list = []
	feat_list = []
	for i in range(batch_size):
		start = i * node_per_pool_graph
		end = (i + 1) * node_per_pool_graph
		adj_list.append(batch_adj[start:end, start:end])
		feat_list.append(batch_feat[start:end, :])
	adj_list = list(map(lambda x: torch.unsqueeze(x, 0), adj_list))
	feat_list = list(map(lambda x: torch.unsqueeze(x, 0), feat_list))
	adj = torch.cat(adj_list, dim=0)
	feat = torch.cat(feat_list, dim=0)

	return feat, adj
def masked_softmax(matrix, mask, dim=-1, memory_efficient=True,
				   mask_fill_value=-1e32):
	'''
	masked_softmax for dgl batch graph
	code snippet contributed by AllenNLP (https://github.com/allenai/allennlp)
	'''
	if mask is None:
		result = torch.nn.functional.softmax(matrix, dim=dim)
	else:
		mask = mask.float()
		while mask.dim() < matrix.dim():
			mask = mask.unsqueeze(1)
		if not memory_efficient:
			result = torch.nn.functional.softmax(matrix * mask, dim=dim)
			result = result * mask
			result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
		else:
			masked_matrix = matrix.masked_fill((1 - mask).byte(),
											   mask_fill_value)
			result = torch.nn.functional.softmax(masked_matrix, dim=dim)
	return result

class BatchedGraphSAGE(nn.Module):
	def __init__(self, infeat, outfeat, use_bn=True,
				 mean=False, add_self=False):
		super().__init__()
		self.add_self = add_self
		self.use_bn = use_bn
		self.mean = mean
		self.W = nn.Linear(infeat, outfeat, bias=True)

		nn.init.xavier_uniform_(
			self.W.weight,
			gain=nn.init.calculate_gain('relu'))

	def forward(self, x, adj):
		num_node_per_graph = adj.size(1)
		if self.use_bn and not hasattr(self, 'bn'):
			self.bn = nn.BatchNorm1d(num_node_per_graph).to(adj.device)

		if self.add_self:
			adj = adj + torch.eye(num_node_per_graph).to(adj.device)

		if self.mean:
			adj = adj / adj.sum(-1, keepdim=True)

		h_k_N = torch.matmul(adj, x)
		h_k = self.W(h_k_N)
		h_k = F.normalize(h_k, dim=2, p=2)
		h_k = F.relu(h_k)
		if self.use_bn:
			h_k = self.bn(h_k)
		return h_k

	def __repr__(self):
		if self.use_bn:
			return 'BN' + super(BatchedGraphSAGE, self).__repr__()
		else:
			return super(BatchedGraphSAGE, self).__repr__()
class Bundler(nn.Module):
	"""
	Bundler, which will be the node_apply function in DGL paradigm
	"""

	def __init__(self, in_feats, out_feats, activation, dropout, bias=True):
		super(Bundler, self).__init__()
		self.dropout = nn.Dropout(p=dropout)
		self.linear = nn.Linear(in_feats * 2, out_feats,bias)
		self.activation = activation

		nn.init.xavier_uniform_(self.linear.weight,
								gain=nn.init.calculate_gain('relu'))

	def concat(self, h, aggre_result):
		bundle = torch.cat((h, aggre_result), 1)
		bundle = self.linear(bundle)
		
		# bundle = self.linear(bundle)
		return bundle

	def forward(self, node):
		h = node.data['h']
		c = node.data['c']
		bundle = self.concat(h, c)
		bundle = F.normalize(bundle, p=2, dim=1)
		if self.activation:
			bundle = self.activation(bundle)
		return {"h": bundle}
class Aggregator(nn.Module):
	"""
	Base Aggregator class. Adapting
	from PR# 403
	This class is not supposed to be called
	"""

	def __init__(self):
		super(Aggregator, self).__init__()

	def forward(self, node):
		neighbour = node.mailbox['m']
		c = self.aggre(neighbour)
		return {"c": c}

	def aggre(self, neighbour):
		# N x F
		raise NotImplementedError
class MeanAggregator(Aggregator):
	'''
	Mean Aggregator for graphsage
	'''

	def __init__(self):
		super(MeanAggregator, self).__init__()

	def aggre(self, neighbour):
		mean_neighbour = torch.mean(neighbour, dim=1)
		return mean_neighbour
class MaxPoolAggregator(Aggregator):
	'''
	Maxpooling aggregator for graphsage
	'''

	def __init__(self, in_feats, out_feats, activation, bias):
		super(MaxPoolAggregator, self).__init__()
		self.linear = nn.Linear(in_feats, out_feats, bias=bias)
		self.activation = activation
		# Xavier initialization of weight
		nn.init.xavier_uniform_(self.linear.weight,
								gain=nn.init.calculate_gain('relu'))

	def aggre(self, neighbour):
		neighbour = self.linear(neighbour)
		if self.activation:
			neighbour = self.activation(neighbour)
		maxpool_neighbour = torch.max(neighbour, dim=1)[0]
		return maxpool_neighbour
class LSTMAggregator(Aggregator):
	'''
	LSTM aggregator for graphsage
	'''

	def __init__(self, in_feats, hidden_feats):
		super(LSTMAggregator, self).__init__()
		self.lstm = nn.LSTM(in_feats, hidden_feats, batch_first=True)
		self.hidden_dim = hidden_feats
		self.hidden = self.init_hidden()

		nn.init.xavier_uniform_(self.lstm.weight,
								gain=nn.init.calculate_gain('relu'))

	def init_hidden(self):
		"""
		Defaulted to initialite all zero
		"""
		return (torch.zeros(1, 1, self.hidden_dim),
				torch.zeros(1, 1, self.hidden_dim))

	def aggre(self, neighbours):
		'''
		aggregation function
		'''
		# N X F
		rand_order = torch.randperm(neighbours.size()[1])
		neighbours = neighbours[:, rand_order, :]

		(lstm_out, self.hidden) = self.lstm(neighbours.view(neighbours.size()[0],
															neighbours.size()[
			1],
			-1))
		return lstm_out[:, -1, :]

	def forward(self, node):
		neighbour = node.mailbox['m']
		c = self.aggre(neighbour)
		return {"c": c}
class GraphSageLayer(nn.Module):
	"""
	GraphSage layer in Inductive learning paper by hamilton
	Here, graphsage layer is a reduced function in DGL framework
	"""

	def __init__(self, in_feats, out_feats, activation, dropout,
				 aggregator_type, bn=False, bias=True):
		super(GraphSageLayer, self).__init__()
		self.use_bn = bn
		self.bundler = Bundler(in_feats, out_feats, activation, dropout,
							   bias=bias)
		self.dropout = nn.Dropout(p=dropout)

		if aggregator_type == "maxpool":
			self.aggregator = MaxPoolAggregator(in_feats, in_feats,
												activation, bias)
		elif aggregator_type == "lstm":
			self.aggregator = LSTMAggregator(in_feats, in_feats)
		else:
			self.aggregator = MeanAggregator()

	def forward(self, g, h):
		h = self.dropout(h)
		g.ndata['h'] = h
		if self.use_bn and not hasattr(self, 'bn'):
			device = h.device
			self.bn = nn.BatchNorm1d(h.size()[1]).to(device)
		g.update_all(fn.copy_src(src='h', out='m'), self.aggregator,
					 self.bundler)
		if self.use_bn:
			h = self.bn(h)
		h = g.ndata.pop('h')
		return h
class GraphSage(nn.Module):
	"""
	Grahpsage network that concatenate several graphsage layer
	"""

	def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation,
				 dropout, aggregator_type):
		super(GraphSage, self).__init__()
		self.layers = nn.ModuleList()

		# input layer
		self.layers.append(GraphSageLayer(in_feats, n_hidden, activation, dropout,
										  aggregator_type))
		# hidden layers
		for _ in range(n_layers - 1):
			self.layers.append(GraphSageLayer(n_hidden, n_hidden, activation,
											  dropout, aggregator_type))
		# output layer
		self.layers.append(GraphSageLayer(n_hidden, n_classes, None,
										  dropout, aggregator_type))

	def forward(self, g, features):
		h = features
		for layer in self.layers:
			h = layer(g, h)
		return h
class DiffPoolBatchedGraphLayer(nn.Module):

	def __init__(self, input_dim, assign_dim, output_feat_dim,
				 activation, dropout, aggregator_type, link_pred):
		super(DiffPoolBatchedGraphLayer, self).__init__()
		self.embedding_dim = input_dim
		self.assign_dim = assign_dim
		self.hidden_dim = output_feat_dim
		self.link_pred = link_pred
		self.feat_gc = GraphSageLayer(
			input_dim,
			output_feat_dim,
			activation,
			dropout,
			aggregator_type)
		self.pool_gc = GraphSageLayer(
			input_dim,
			assign_dim,
			activation,
			dropout,
			aggregator_type)

	def forward(self, g, h):
		feat = self.feat_gc(g, h)  # size = (sum_N, F_out), sum_N is num of nodes in this batch
		device = feat.device
		# assign_tensor = self.pool_gc(g,h)  # size = (sum_N, N_a), N_a is num of nodes in pooled graph.
		assign_tensor = self.pool_gc(g, h)
		assign_tensor = torch.exp(F.log_softmax(assign_tensor, dim=1))
		# print(assign_tensor.shape)
		assign_tensor = torch.split(assign_tensor, g.batch_num_nodes().tolist())
		assign_tensor = torch.block_diag(*assign_tensor)  # size = (sum_N, batch_size * N_a)

		h = torch.matmul(torch.t(assign_tensor), feat)
		adj = g.adjacency_matrix(transpose=False, ctx=device)
		adj_new = torch.sparse.mm(adj, assign_tensor)
		adj_new = torch.mm(torch.t(assign_tensor), adj_new)


		return adj_new, h, assign_tensor

class ClassificationOutputModule(MetaModule):
    def __init__(self, node_embedding_dim, num_classes):
        super(ClassificationOutputModule, self).__init__()
        self.linear = MetaLinear(node_embedding_dim, num_classes)
        
    def forward(self, inputs, params=None):
        x = self.linear(inputs, params=self.get_subdict(params, 'linear'))
        return x
class gClassificationOutputModule(MetaModule):
    def __init__(self, node_embedding_dim, num_classes):
        super(gClassificationOutputModule, self).__init__()
        self.linear1 = MetaLinear(node_embedding_dim, node_embedding_dim)
        self.linear2 = MetaLinear(node_embedding_dim, num_classes)
        
    def forward(self, inputs, params=None):
        x = self.linear1(inputs, params=self.get_subdict(params, 'linear'))
        x=F.relu(x)
        x = self.linear2(x, params=self.get_subdict(params, 'linear'))
        return x
class MetaOutputLayers(MetaModule):
	def __init__(self, node_embedding_dim, nc_num_classes, gc_num_classes):
		super(MetaOutputLayers, self).__init__()
		
		### try nn.ModuleDict
		self.nc_0_output_layer = ClassificationOutputModule(node_embedding_dim, 1)
		self.nc_1_output_layer = ClassificationOutputModule(64, 1)
		self.cc_output_layer = ClassificationOutputModule(3, nc_num_classes)
		self.gc_0_output_layer = gClassificationOutputModule(2*node_embedding_dim, 1)
		self.gc_1_output_layer = gClassificationOutputModule(2*node_embedding_dim, 2)
		self.bc = ClassificationOutputModule(node_embedding_dim, nc_num_classes)
	
	def forward(self, node_embs, inputs, task_selector, params):
		if task_selector == "nc_0":
			x = self.nc_0_output_layer(node_embs, params=self.get_subdict(params, 'nc_0_output_layer'))
		elif task_selector == "nc_1":
			x = self.nc_1_output_layer(node_embs, params=self.get_subdict(params, 'nc_1_output_layer'))
		elif task_selector == "cc":
			x = self.cc_output_layer(node_embs, params=self.get_subdict(params, 'cc_output_layer'))
		elif task_selector == "gc_0":
			x = self.gc_0_output_layer(node_embs, 
									 params=self.get_subdict(params, 'gc_0_output_layer'))
		elif task_selector == "gc_1":
			x = self.gc_1_output_layer(node_embs, 
									 params=self.get_subdict(params, 'gc_1_output_layer'))
		elif task_selector == "bc":
			x = self.bc_output_layer(node_embs, 
									 params=self.get_subdict(params, 'bc_output_layer'))
		else:
			print("Invalid task selector.")
		
		return x
class DiffPool(MetaModule):
	"""
	DiffPool Fuse
	"""

	def __init__(self, input_dim, hidden_dim, embedding_dim,
				 label_dim, activation, n_layers, dropout,
				 n_pooling, linkpred, batch_size, aggregator_type,
				 assign_dim, cat=False):
		super(DiffPool, self).__init__()
		self.fcc1 = nn.Linear(1, 4)
		self.fcc2_1 = nn.Linear(772, hidden_dim)
		self.fcc2_2 = nn.Linear(hidden_dim, hidden_dim)
		self.uemb1 = nn.Linear(hidden_dim,hidden_dim)
		
		# self.com_vul_fc = nn.Linear(3, 3)
		self.link_pred = linkpred
		self.concat = cat
		self.n_pooling = n_pooling
		self.batch_size = batch_size
		self.gc_before_pool = nn.ModuleList()
		# self.link_pred_loss = []
		# self.entropy_loss = []

		# list of GNN modules before the first diffpool operation
		self.diffpool_layers = nn.ModuleList()
		self.node_ = nn.ModuleList()

		# list of list of GNN modules, each list after one diffpool operation
		self.gc_after_pool = nn.ModuleList()
		self.assign_dim = assign_dim
		self.bn = True
		self.num_aggs = 1
		self.weights = torch.nn.Parameter(torch.ones(3).float())
		

		self.user_gc = GraphSageLayer(
			2*hidden_dim,
			hidden_dim,
			activation,
			dropout,
			aggregator_type)

		assign_dims = []
		assign_dims.append(self.assign_dim)
		pool_embedding_dim = embedding_dim
		self.first_diffpool_layer = DiffPoolBatchedGraphLayer(
			pool_embedding_dim,
			self.assign_dim,
			hidden_dim,
			activation,
			dropout,
			aggregator_type,
			self.link_pred)
		gc_after_per_pool = nn.ModuleList()

		gc_after_per_pool.append(BatchedGraphSAGE(hidden_dim, embedding_dim))
		self.gc_after_pool.append(gc_after_per_pool)
		self.output_layer = MetaOutputLayers(embedding_dim, 3, 4)

	def gcn_forward(self, g, h, gc_layers, cat=False):
		"""
		Return gc_layer embedding cat.
		"""
		block_readout = []
		for gc_layer in gc_layers[:-1]:
			h = gc_layer(g, h)
			block_readout.append(h)
		h = gc_layers[-1](g, h)
		block_readout.append(h)
		if cat:
			block = torch.cat(block_readout, dim=1)  # N x F, F = F1 + F2 + ...
		else:
			block = h
		return block

	def gcn_forward_tensorized(self, h, adj, gc_layers, cat=False):
		block_readout = []
		for gc_layer in gc_layers:
			h = gc_layer(h, adj)
			block_readout.append(h)
		if cat:
			block = torch.cat(block_readout, dim=2)  # N x F, F = F1 + F2 + ...
		else:
			block = h
		return block

	def forward(self, g, inputs, t, uemb, params=None):

		t = torch.reshape(t, (-1,1))
		h_2 = self.fcc1(t)
		h = torch.cat((h_2, inputs), 1)
		
		# h = h_2
		out_all = []
		
		# g_embedding = self.gcn_forward(g, h, self.gc_before_pool, self.concat)
		g_embedding = F.relu(self.fcc2_1(h))
		uemb = F.relu(self.fcc2_2(uemb))


		q = g_embedding
		alpha = (q*uemb).sum(-1).reshape(-1,1)
		alpha = torch.exp(F.log_softmax(alpha,-1))
		g_embedding = alpha*uemb

		
		# g_embedding = torch.cat((g_embedding, uemb), 1)
		g.ndata['h'] = g_embedding
		adj, h, s = self.first_diffpool_layer(g, g_embedding)

		node_per_pool_graph = int(adj.size()[0] / len(g.batch_num_nodes()))
		
		h, adj = batch2tensor(adj, h, node_per_pool_graph)
		h = self.gcn_forward_tensorized(
			h, adj, self.gc_after_pool[0], self.concat)

		# ypred = torch.exp(F.log_softmax(ypred,-1))
		com_vul = torch.matmul(s, h.reshape(-1,h.shape[-1]))
		node_rep = torch.cat((g_embedding,com_vul),1)
		node_rep = self.user_gc(g, node_rep)
		final_h = self.output_layer(node_rep, inputs, 'nc_1', params=self.get_subdict(params, 'output_layer'))

		readout = torch.mean(h, dim=1)
		# final_readout = readout
		with g.local_scope():
			g.ndata['h'] = node_rep
			hg = dgl.max_nodes(g, 'h')
		final_readout = torch.cat((readout,hg),1)

		ypred0 = self.output_layer(final_readout, inputs, 'gc_0', params=self.get_subdict(params, 'output_layer'))
		ypred1 = self.output_layer(final_readout, inputs, 'gc_1', params=self.get_subdict(params, 'output_layer'))
		return ypred1, final_h, final_h, ypred0

