import dgl
import torch
from torchmetrics import RetrievalMRR
from sklearn.metrics import  accuracy_score, precision_score, recall_score, f1_score, ndcg_score
def evaluation_1(prediction, y): 
	loss = torch.nn.MSELoss()
	mrr = RetrievalMRR()
	return loss(prediction,y), MSLELoss(prediction,y), ndcg_score(prediction.reshape(1,-1),y.reshape(1,-1)), 0
def evaluation_2(prediction, y):
	p = prediction.topk(1)[1].reshape(-1).cpu()
	return accuracy_score(y,p), precision_score(y,p), recall_score(y,p), f1_score(y,p,labels=[0,1],average='macro',zero_division = 1)
def MSLELoss(pred, actual):
	loss = torch.nn.MSELoss()
	return loss(torch.log(pred + 1), torch.log(actual + 1))

