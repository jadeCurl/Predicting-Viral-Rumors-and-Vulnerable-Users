# from model_upfd import *
from model import *
from utils import *
def _collate_fn(batch):
	graphs = [entry[0] for entry in batch]
	veracity_labels = [entry[1] for entry in batch]
	virality_labels = [entry[2] for entry in batch]
	#graphs, labels = batch
	g = dgl.batch(graphs)
	veracity_l = torch.tensor(veracity_labels, dtype=torch.long)
	virality_l = torch.tensor(virality_labels, dtype=torch.int)
	return g, veracity_l, virality_l
def train_func(sub_train_, epoch, ini,model,criterion1,criterion2,optimizer,scheduler,bs,lam,device,step):
	train_loss_g_vir = 0.0
	train_loss_g_ver = 0.0
	train_loss_n = 0.0
	train_acc_g_vir = 0.0
	train_acc_g_ver = 0.0
	train_acc_n = 0.0

	model.train()

	data = DataLoader(sub_train_, batch_size=bs, shuffle=True, collate_fn=_collate_fn, drop_last=True)
	for i, (graph, veracity_label, virality_label) in enumerate(data):

		optimizer.zero_grad()

		graph, veracity_label, virality_label = graph.to(device), veracity_label.to(device), virality_label.to(device)
		label_n = graph.ndata['y']
		
		output = model(graph, graph.ndata['x'], graph.ndata['t'], graph.ndata['emb']) #55226*300
		index_l_0 = torch.nonzero(label_n != -1)
		index_l_0 = torch.reshape(index_l_0, (-1,))
		label_n_0 = torch.index_select(label_n, 0, index_l_0)
		output_1_0 = torch.index_select(output[1], 0, index_l_0)
		output_2_0 = torch.index_select(output[2], 0, index_l_0)

		for task in ['ver_c','vir_c','nc']:
			if task == 'ver_c': 
				model.zero_grad()
				loss_g_ver = criterion2(output[0], veracity_label).to(device)
				adapted_params = update_parameters_gd(model, loss_g_ver,step_size=step, first_order=None)
				loss_g_ver = criterion2(output[0], veracity_label).to(device)
			elif task == 'vir_c': 
				model.zero_grad()
				virality_label = virality_label.float()
				loss_g_vir = criterion1(output[3].reshape(-1), virality_label).to(device)
				# print(loss_g_vir)
				adapted_params = update_parameters_gd(model, loss_g_vir,step_size=step, first_order=None)
				loss_g_vir = criterion1(output[3].reshape(-1), virality_label).to(device)
			elif task == 'nc':	
				model.zero_grad()	
				loss_n = criterion1(output_1_0.reshape(-1), label_n_0) 
				loss_n_2 = criterion1(output_2_0.reshape(-1), label_n_0)  
				loss_n = (1-lam)*loss_n + lam*loss_n_2
				adapted_params = update_parameters_gd(model, loss_n,step_size=step, first_order=None)
				loss_n = criterion1(output_1_0.reshape(-1), label_n_0) 
				loss_n_2 = criterion1(output_2_0.reshape(-1), label_n_0) 
				loss_n = (1-lam)*loss_n + lam*loss_n_2
		train_loss_g_vir += loss_g_vir.item()
		train_loss_g_ver += loss_g_ver.item()
		train_loss_n += loss_n.item()
		loss = loss_g_vir + loss_g_ver + loss_n 
		loss.backward(retain_graph=True)
		
		optimizer.step()			
		train_acc_g_vir += MSLELoss(output[3].reshape(-1), virality_label)
		train_acc_g_ver += (output[0].argmax(1) == veracity_label).sum().item()
		if (output_1_0.shape[0] != 0):
			train_acc_n += MSLELoss(output_1_0.reshape(-1),label_n_0) / output_1_0.shape[0]

		del graph
		del veracity_label
		del virality_label
		torch.cuda.empty_cache()

	# Adjust the learning rate
	scheduler.step()
	# scheduler_warmup.step(epoch)
	return ini, train_loss_g_ver / len(data), train_loss_g_vir / len(data), train_loss_n / len(data), train_acc_g_ver / len(sub_train_), train_acc_g_vir / len(sub_train_),train_acc_n / len(data)
def ft_func(sub_train_, epoch, ini,model,criterion1,criterion2,optimizer,scheduler, bs, lam,device,step):
	train_loss_g_vir = 0.0
	train_loss_g_ver = 0.0
	train_loss_n = 0.0
	train_acc_g_vir = 0.0
	train_acc_g_ver = 0.0
	train_acc_n = 0.0
	model.train()

	data = DataLoader(sub_train_, batch_size=bs, shuffle=True, collate_fn=_collate_fn, drop_last=True)
	for i, (graph, veracity_label, virality_label) in enumerate(data):

		optimizer.zero_grad()

		graph, veracity_label, virality_label = graph.to(device), veracity_label.to(device), virality_label.to(device)
		label_n = graph.ndata['y']
		# label_n = label_n.long()
		
		output = model(graph, graph.ndata['x'], graph.ndata['t'], graph.ndata['emb']) #55226*300
		index_l_0 = torch.nonzero(label_n != -1)
		index_l_0 = torch.reshape(index_l_0, (-1,))
		label_n_0 = torch.index_select(label_n, 0, index_l_0)
		output_1_0 = torch.index_select(output[1], 0, index_l_0)
		output_2_0 = torch.index_select(output[2], 0, index_l_0)
		

		for task in ['ver_c','vir_c','nc']:
			if task == 'ver_c': 
				model.zero_grad()
				loss_g_ver = criterion2(output[0], veracity_label).to(device)
				adapted_params = update_parameters_gd(model, loss_g_ver,step_size=step, first_order=None)
				loss_g_ver = criterion2(output[0], veracity_label).to(device)
			elif task == 'vir_c': 
				model.zero_grad()
				virality_label = virality_label.float()
				loss_g_vir = criterion1(output[3].reshape(-1), virality_label).to(device)
				# print(loss_g_vir)
				adapted_params = update_parameters_gd(model, loss_g_vir,step_size=step, first_order=None)
				loss_g_vir = criterion1(output[3].reshape(-1), virality_label).to(device)
			elif task == 'nc':	
				model.zero_grad()	
				loss_n = criterion1(output_1_0.reshape(-1), label_n_0) #+ entropy(output_1_1, 0)
				# loss_n = loss_n.to(device)
				loss_n_2 = criterion1(output_2_0.reshape(-1), label_n_0)  #+ entropy(output_1_1, 0)
				# loss_n_2 = loss_n_2.to(device)
				loss_n = (1-lam)*loss_n + lam*loss_n_2
				adapted_params = update_parameters_gd(model, loss_n,step_size=step, first_order=None)
				loss_n = criterion1(output_1_0.reshape(-1), label_n_0) #+ entropy(output_1_1, 0)
				# loss_n = loss_n.to(device)
				loss_n_2 = criterion1(output_2_0.reshape(-1), label_n_0)  #+ entropy(output_1_1, 0)
				# loss_n_2 = loss_n_2.to(device)
				loss_n = (1-lam)*loss_n + lam*loss_n_2
		train_loss_g_vir += loss_g_vir.item()
		train_loss_g_ver += loss_g_ver.item()
		train_loss_n += loss_n.item()
		loss = loss_g_vir + loss_g_ver + loss_n 			
		train_acc_g_vir += MSLELoss(output[3].reshape(-1), virality_label)
		train_acc_g_ver += (output[0].argmax(1) == veracity_label).sum().item()
		if (output_1_0.shape[0] != 0):
			train_acc_n += MSLELoss(output_1_0.reshape(-1),label_n_0) / output_1_0.shape[0]

		del graph
		del veracity_label
		del virality_label
		torch.cuda.empty_cache()

	# Adjust the learning rate
	# scheduler.step()
	# scheduler_warmup.step(epoch)
	return ini, train_loss_g_ver / len(data), train_loss_g_vir / len(data), train_loss_n / len(data), train_acc_g_ver / len(sub_train_), train_acc_g_vir / len(sub_train_),train_acc_n / len(data)
def test(data_,model,criterion1,criterion2,optimizer,scheduler,bs,device):
	total_dic = {}
	num_1 = {}
	num_m = {}
	label_all = {}
	label_1 = {}
	label_m = {}

	evaluate_g_ver  = np.zeros(2)
	evaluate_g_vir  = np.zeros(2)
	evaluate_u = np.zeros(2)

	model.eval()

	data = DataLoader(data_, batch_size=bs, shuffle=True, collate_fn=_collate_fn, drop_last=True)
	og_ver = torch.tensor([])
	lg_ver = []
	og_vir = torch.tensor([])
	lg_vir = []
	on = torch.tensor([])
	ln = []
	for graph, veracity_label, virality_label in data:
		graph= graph.to(device)

		label_n = graph.ndata['y']

		with torch.no_grad():
			output = model(graph, graph.ndata['x'], graph.ndata['t'], graph.ndata['emb'])
			og_ver = torch.cat((og_ver,output[0].cpu().detach()),0)
			lg_ver.extend(veracity_label.cpu().detach().numpy().tolist())
			og_vir = torch.cat((og_vir,output[3].cpu().detach()),0)
			lg_vir.extend(virality_label.float().cpu().detach().numpy().tolist())
			
			index_l_0 = torch.nonzero(label_n != -1)
			index_l_0 = torch.reshape(index_l_0, (-1,))
			label_n_0 = torch.index_select(label_n, 0, index_l_0)
			output_1_0 = torch.index_select(output[1], 0, index_l_0)
			output_2_0 = torch.index_select(output[2], 0, index_l_0)
			
			on = torch.cat((on,output_1_0.cpu().detach()),0)
			ln.extend(label_n_0.cpu().detach().numpy().tolist())			
		
			del graph
			del veracity_label
			del virality_label
			torch.cuda.empty_cache()
	evaluate_g_ver = evaluation_2(og_ver,lg_ver)
	evaluate_g_vir = evaluation_1(og_vir.reshape(-1),torch.tensor(lg_vir))
	evaluate_u = evaluation_1(on.reshape(-1), torch.tensor(ln)) 
	return evaluate_g_ver, evaluate_g_vir, evaluate_u#, evaluate_u1, evaluate_um