import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import dgl.function as fn
from dgl.data import *
from dgl.data.utils import save_graphs
import random
import csv
import numpy as np
from dgl.data.utils import load_graphs
from dgl.data import DGLDataset
class Weibo(DGLDataset):
	def __init__(self,raw_dir=None,force_reload=False,verbose=False):
		super(Weibo, self).__init__(name='Weibo', raw_dir=raw_dir, force_reload=force_reload, verbose=verbose)

	def process(self):
		mat_path = self.raw_path[0:-6]
		# 将数据处理为图列表和标签列表
		self.graphs, self.label = load_graphs(mat_path)
		self.veracity_label = list(self.label['graph_veracity'].numpy())
		self.virality_label = list(np.log2(self.label['graph_sizes'].numpy()+1))
		# self.virality_label = list(self.label['graph_sizes'].numpy())
		# self.label = list(self.label.values())[0].numpy()

	def __getitem__(self, idx):
		graph = self.graphs[idx]
		veracity_label = self.veracity_label[idx]
		virality_label = self.virality_label[idx]
		graph.ndata['emb'] = graph.ndata['emb'].float()
		graph.ndata['z'] = torch.randn(graph.ndata['x'].shape[0],128)
		graph.ndata['y'] = graph.ndata['y'].to(torch.float)
		
		return graph, veracity_label, virality_label
	def __len__(self):
		"""数据集中图的数量"""
		return len(self.graphs)