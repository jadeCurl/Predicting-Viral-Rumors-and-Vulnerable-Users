from model import *
from run_func import *
def setup_seed(seed):
	 torch.manual_seed(seed)
	 torch.cuda.manual_seed_all(seed)
	 np.random.seed(seed)
	 random.seed(seed)
	 torch.backends.cudnn.deterministic = True
	 
if __name__ == "__main__":
	seed_=42
	setup_seed(seed_)
	parser = argparse.ArgumentParser(description='change the parameter.')
	parser.add_argument("-e", '--time', type=float, help='an integer')
	parser.add_argument('--no_link_pred', dest='linkpred', action='store_false',
							help='switch of link prediction object')

	args = vars(parser.parse_args())

	t = args["time"]

	path_1 = './dataset/Twitter/train_' + str(t)
	path_2 = './dataset/Twitter/valid_' + str(t)
	path_3 = './dataset/Twitter/test_' + str(t)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	sub_train_ = Weibo(path_1)
	sub_valid_ = Weibo(path_2)
	sub_test_ = Weibo(path_3)

	bs =2
	dp = 0.2
	step = 0.4
	node_features = 64
	hid_features = 64
	out_features = 64
	activation = F.relu
	pool_size = 50
	model = DiffPool(node_features, hid_features, out_features, 4, activation, 2, dp, 1, parser.parse_args().linkpred, bs,'meanpool', pool_size)
	model.to(device)

	
	lam = 0

	import time
	from torch.utils.data.dataset import random_split
	max_epoch = 30
	min_valid_loss = float('inf')
	criterion1 = torch.nn.MSELoss().to(device)
	criterion2 = torch.nn.CrossEntropyLoss().to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.8)

	for epoch in range(max_epoch):

		start_time = time.time()
		if epoch == 0:
			initial_task_loss = 0
		initial_task_loss, train_loss_g_ver, train_loss_g_vir, train_loss_n, train_acc_g_ver, train_acc_g_vir, train_acc_n = train_func(sub_train_, epoch, initial_task_loss,model,criterion1,criterion2,optimizer,scheduler,bs,lam,device,step)
		initial_task_loss, train_loss_g_ver, train_loss_g_vir, train_loss_n, train_acc_g_ver, train_acc_g_vir, train_acc_n = ft_func(sub_train_, epoch, initial_task_loss,model,criterion1,criterion2,optimizer,scheduler,bs,lam,device,step)
		
		evaluate_g_ver, evaluate_g_vir, evaluate_u = test(sub_valid_,model,criterion1,criterion2,optimizer,scheduler,bs,device)
		evaluate_test_g_ver, evaluate_test_g_vir, evaluate_test_u = test(sub_test_,model,criterion1,criterion2,optimizer,scheduler,bs,device)

		secs = int(time.time() - start_time)
		mins = secs / 60
		secs = secs % 60
										
		print('Epoch: %d' %(epoch + 1), " | time in %d minutes, %d seconds" %(mins, secs))
		print(f'\tLoss: {train_loss_g_ver:.4f}(train_graph)\t|\tAcc: {train_acc_g_ver * 100:.1f}%(train_graph)')
		print(f'\tLoss: {train_loss_g_vir:.4f}(train_graph)\t|\tmlse: {train_acc_g_vir:.4f}(train_graph)')
		print(f'\tLoss: {train_loss_n:.4f}(train_node)\t|\tmlse: {train_acc_n:.4f}(train_node)')
		print(f'\tAcc: {evaluate_test_g_ver[0]:.3f}\tPrecision: {evaluate_test_g_ver[1]:.3f}\tRecall: {evaluate_test_g_ver[2]:.3f}\tF1: {evaluate_test_g_ver[3]:.3f}')
		print(f'\tMSE: {evaluate_test_g_vir[0]:.3f}\tMLSE: {evaluate_test_g_vir[1]:.3f}\tNDCG: {evaluate_test_g_vir[2]:.3f}\tMRR: {evaluate_test_g_vir[3]:.3f}')
		print(f'\tMSE: {evaluate_test_u[0]:.3f}\tMLSE: {evaluate_test_u[1]:.3f}\tNDCG: {evaluate_test_u[2]:.3f}\tMRR: {evaluate_test_u[3]:.3f}')
