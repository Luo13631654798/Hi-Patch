import os
import sys
sys.path.append("..")
import time
import datetime
import argparse
import numpy as np

from random import SystemRandom

parser = argparse.ArgumentParser('ITS Forecasting')



parser.add_argument('--state', type=str, default='def')
parser.add_argument('-n',  type=int, default=int(1e8), help="Size of the dataset")
parser.add_argument('--epoch', type=int, default=1000, help="training epoches")
parser.add_argument('--patience', type=int, default=10, help="patience for early stop")
parser.add_argument('--history', type=int, default=24, help="number of hours (months for ushcn and ms for activity) as historical window")
parser.add_argument('--pred_window', type=int, default=1, help="number of hours (months for ushcn) as pred window")
parser.add_argument('--logmode', type=str, default="a", help='File mode of logging.')
parser.add_argument('--lr',  type=float, default=1e-3, help="Starting learning rate.")
parser.add_argument('--w_decay', type=float, default=0.0, help="weight decay.")
parser.add_argument('-b', '--batch_size', type=int, default=32)
parser.add_argument('--load', type=str, default=None, help="ID of the experiment to load for evaluation. If None, run a new experiment.")
parser.add_argument('--seed', type=int, default=1, help="Random seed")
parser.add_argument('--dataset', type=str, default='physionet', help="Dataset to load. Available: physionet, mimic, ushcn, activity")
# value 0 means using original time granularity, Value 1 means quantization by 1 hour, 
# value 0.1 means quantization by 0.1 hour = 6 min, value 0.016 means quantization by 0.016 hour = 1 min
parser.add_argument('--quantization', type=float, default=0.0, help="Quantization on the physionet dataset.")
parser.add_argument('--model', type=str, default='Hi-Patch', help="Model name")
parser.add_argument('--nhead', type=int, default=1, help="heads in Transformer")
parser.add_argument('--nlayer', type=int, default=1, help="# of layer in TSmodel")
parser.add_argument('-ps', '--patch_size', type=float, default=24, help="window size for a patch")
parser.add_argument('--stride', type=float, default=24, help="period stride for patch sliding")
parser.add_argument('-hd', '--hid_dim', type=int, default=64, help="Number of units per hidden layer")
parser.add_argument('--alpha', type=float, default=0.9, help="Uncertainty base number")
parser.add_argument('--res', type=float, default=1, help="Res")
parser.add_argument('--gpu', type=str, default='0', help='which gpu to use.')

args = parser.parse_args()
args.npatch = int(np.ceil((args.history - args.patch_size) / args.stride)) + 1 # (window size for a patch)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import torch
import torch.optim as optim
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic=True
torch.use_deterministic_algorithms(True)
# from lib.plotting import *
import lib.utils as utils
from lib.parse_datasets import parse_datasets
from lib.evaluation import *
from model.hipatch import *
import warnings
warnings.filterwarnings("ignore")

file_name = os.path.basename(__file__)[:-3]
args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args.PID = os.getpid()

print("PID, device:", args.PID, args.device)

#####################################################################################################
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def layer_of_patches(n_patch):
    if n_patch == 1:
        return 1
    if n_patch % 2 == 0:
        return 1 + layer_of_patches(n_patch / 2)
    else:
        return layer_of_patches(n_patch + 1)

if __name__ == '__main__':
	utils.setup_seed(args.seed)

	experimentID = args.load
	if experimentID is None:
		# Make a new experiment ID
		experimentID = int(SystemRandom().random()*100000)

	input_command = sys.argv
	ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
	if len(ind) == 1:
		ind = ind[0]
		input_command = input_command[:ind] + input_command[(ind+2):]
	input_command = " ".join(input_command)

	# utils.makedirs("results/")

	##################################################################
	data_obj = parse_datasets(args, patch_ts=True)
	input_dim = data_obj["input_dim"]
	
	### Model setting ###
	args.ndim = input_dim
	args.npatch = int(math.ceil((args.history - args.patch_size) / args.stride)) + 1
	args.patch_layer = layer_of_patches(args.npatch)
	args.scale_patch_size = args.patch_size / (args.history + args.pred_window)
	args.task = 'forecasting'

	model = Hi_Patch(args).to(args.device)



	params = (list(model.parameters()))
	print('model', model)
	print('parameters:', count_parameters(model))

	##################################################################

	if(args.n < 12000):
		args.state = "debug"
		log_path = "logs/{}_{}_{}.log".format(args.dataset, args.model, args.state)
	else:
		log_path = "logs/{}_{}_{}_{}patch_{}stride_{}layer_{}lr_{}seed.log". \
			format(args.dataset, args.model, args.state, args.patch_size, args.stride, args.nlayer, args.lr, args.seed)
	
	if not os.path.exists("logs/"):
		utils.makedirs("logs/")
	logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__), mode=args.logmode)
	logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
	logger.info(input_command)
	logger.info(args)

	optimizer = optim.Adam(model.parameters(), lr=args.lr)

	num_batches = data_obj["n_train_batches"] # n_sample / batch_size
	print("n_train_batches:", num_batches)

	best_val_mse = np.inf
	test_res = None
	for itr in range(args.epoch):
		st = time.time()

		### Training ###
		model.train()
		for _ in range(num_batches):
			optimizer.zero_grad()
			batch_dict = utils.get_next_batch(data_obj["train_dataloader"])
			# print(batch_dict["tp_to_predict"].shape, batch_dict["observed_data"].shape, \
		 	# 	batch_dict["observed_tp"].shape, batch_dict["observed_mask"].shape)
			train_res = compute_all_losses(model, batch_dict)
			train_res["loss"].backward()
			optimizer.step()

		### Validation ###
		model.eval()
		with torch.no_grad():
			val_res = evaluation(model, data_obj["val_dataloader"], data_obj["n_val_batches"])
			
			### Testing ###
			# if(val_res["mse"] < best_val_mse):
			# 	best_val_mse = val_res["mse"]
			if (val_res["mse"] < best_val_mse):
				best_val_mse = val_res["mse"]
				best_iter = itr
				test_res = evaluation(model, data_obj["test_dataloader"], data_obj["n_test_batches"])
			
			logger.info('- Epoch {:03d}, ExpID {}'.format(itr, experimentID))
			logger.info("Train - Loss (one batch): {:.5f}".format(train_res["loss"].item()))
			logger.info("Val - Loss, MSE, RMSE, MAE, MAPE: {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.2f}%" \
				.format(val_res["loss"], val_res["mse"], val_res["rmse"], val_res["mae"], val_res["mape"]*100))
			if(test_res != None):
				logger.info("Test - Best epoch, Loss, MSE, RMSE, MAE, MAPE: {}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.2f}%" \
					.format(best_iter, test_res["loss"], test_res["mse"],\
			 		 test_res["rmse"], test_res["mae"], test_res["mape"]*100))
			logger.info("Time spent: {:.2f}s".format(time.time()-st))

		if(itr - best_iter >= args.patience):
			print("Exp has been early stopped!")
			sys.exit(0)