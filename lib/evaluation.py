import torch
import lib.utils as utils

def compute_error(truth, pred_y, mask, func, reduce, norm_dict=None):
	# If pred_y has only 3 dimensions, add an extra dimension
	if len(pred_y.shape) == 3:
		pred_y = pred_y.unsqueeze(dim=0)

	# Get the sizes of predicted data
	n_traj_samples, n_batch, n_tp, n_dim = pred_y.size()

	# Repeat truth and mask to match the shape of pred_y
	truth_repeated = truth.repeat(pred_y.size(0), 1, 1, 1)
	mask = mask.repeat(pred_y.size(0), 1, 1, 1)

	# Calculate error based on the specified function (MSE, MAE, or MAPE)
	if(func == "MSE"):
		error = ((truth_repeated - pred_y)**2) * mask
	elif(func == "MAE"):
		error = torch.abs(truth_repeated - pred_y) * mask
	elif(func == "MAPE"):
		if(norm_dict == None):
			mask = (truth_repeated != 0) * mask
			truth_div = truth_repeated + (truth_repeated == 0) * 1e-8
			error = torch.abs(truth_repeated - pred_y) / truth_div * mask
		else:
			data_max = norm_dict["data_max"]
			data_min = norm_dict["data_min"]
			truth_rescale = truth_repeated * (data_max - data_min) + data_min
			pred_y_rescale = pred_y * (data_max - data_min) + data_min
			mask = (truth_rescale != 0) * mask
			truth_rescale_div = truth_rescale + (truth_rescale == 0) * 1e-8
			error = torch.abs(truth_rescale - pred_y_rescale) / truth_rescale_div * mask
	else:
		raise Exception("Error function not specified")

	# Sum error values along the time and batch dimensions, per variable
	error_var_sum = error.reshape(-1, n_dim).sum(dim=0)
	# Sum the corresponding mask values to count the valid entries
	mask_count = mask.reshape(-1, n_dim).sum(dim=0)

	# Return either mean or sum based on the 'reduce' argument
	if(reduce == "mean"):
		# Calculate average error per variable
		error_var_avg = error_var_sum / (mask_count + 1e-8)
		n_avai_var = torch.count_nonzero(mask_count)
		error_avg = error_var_avg.sum() / n_avai_var
		return error_avg
	elif(reduce == "sum"):
		return error_var_sum, mask_count
	else:
		raise Exception("Reduce argument not specified!")

def compute_all_losses(model, batch_dict):
	# Generate predictions for the specified time points using the model
	pred_y = model.forecasting(batch_dict["tp_to_predict"],
		batch_dict["observed_data"], batch_dict["observed_tp"],
		batch_dict["observed_mask"])

	mse = compute_error(batch_dict["data_to_predict"], pred_y, mask = batch_dict["mask_predicted_data"], func="MSE", reduce="mean")
	rmse = torch.sqrt(mse)

	# Use MSE as the loss function
	mae = compute_error(batch_dict["data_to_predict"], pred_y, mask = batch_dict["mask_predicted_data"], func="MAE", reduce="mean")

	################################
	loss = mse
	# Store the loss and error metrics
	results = {}
	results["loss"] = loss
	results["mse"] = mse.item()
	results["rmse"] = rmse.item()
	results["mae"] = mae.item()

	return results

def evaluation(model, dataloader, n_batches):
	# Initialize result containers
	n_eval_samples = 0
	n_eval_samples_mape = 0
	total_results = {}
	total_results["loss"] = 0
	total_results["mse"] = 0
	total_results["mae"] = 0
	total_results["rmse"] = 0
	total_results["mape"] = 0

	# Loop through batches to evaluate the model
	for _ in range(n_batches):
		# Get the next batch of data from the dataloader
		batch_dict = utils.get_next_batch(dataloader)

		# Predict future time points using the model
		pred_y = model.forecasting(batch_dict["tp_to_predict"],
								   batch_dict["observed_data"], batch_dict["observed_tp"],
								   batch_dict["observed_mask"])

		# Compute Sum of Squared Errors (MSE), Absolute Errors (MAE), and Absolute Percentage Errors (MAPE)
		se_var_sum, mask_count = compute_error(batch_dict["data_to_predict"], pred_y,
											   mask=batch_dict["mask_predicted_data"], func="MSE",
											   reduce="sum")

		ae_var_sum, _ = compute_error(batch_dict["data_to_predict"], pred_y, mask=batch_dict["mask_predicted_data"],
									  func="MAE", reduce="sum")

		ape_var_sum, mask_count_mape = compute_error(batch_dict["data_to_predict"], pred_y,
													 mask=batch_dict["mask_predicted_data"], func="MAPE",
													 reduce="sum")

		# Aggregate errors for all batches
		total_results["loss"] += se_var_sum
		total_results["mse"] += se_var_sum
		total_results["mae"] += ae_var_sum
		total_results["mape"] += ape_var_sum
		n_eval_samples += mask_count
		n_eval_samples_mape += mask_count_mape

	# Compute final averaged errors across all batches
	n_avai_var = torch.count_nonzero(n_eval_samples)
	n_avai_var_mape = torch.count_nonzero(n_eval_samples_mape)

	total_results["loss"] = (total_results["loss"] / (n_eval_samples + 1e-8)).sum() / n_avai_var
	total_results["mse"] = (total_results["mse"] / (n_eval_samples + 1e-8)).sum() / n_avai_var
	total_results["mae"] = (total_results["mae"] / (n_eval_samples + 1e-8)).sum() / n_avai_var
	total_results["rmse"] = torch.sqrt(total_results["mse"])
	total_results["mape"] = (total_results["mape"] / (n_eval_samples_mape + 1e-8)).sum() / n_avai_var_mape

	# Convert tensors to scalar values
	for key, var in total_results.items():
		if isinstance(var, torch.Tensor):
			var = var.item()
		total_results[key] = var

	return total_results