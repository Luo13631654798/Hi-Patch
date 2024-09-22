from lib.physionet import *
from lib.ushcn import *
from lib.mimic import MIMIC
from lib.person_activity import *
from sklearn import model_selection



#####################################################################################################
def parse_datasets(args, patch_ts=False, length_stat=False):

	device = args.device
	dataset_name = args.dataset

	##################################################################
	### PhysioNet dataset ### 
	### MIMIC dataset ###
	if dataset_name in ["physionet", "mimic"]:
		args.pred_window = 48 - args.history
		### list of tuples (record_id, tt, vals, mask) ###
		if dataset_name == "physionet":
			total_dataset = PhysioNet('../data/physionet', quantization = args.quantization,
											download=True, n_samples = args.n, device = device)
		elif dataset_name == "mimic":
			total_dataset = MIMIC('../data/mimic/', n_samples = args.n, device = device)

		if dataset_name == "physionet":
			max_vars_per_plot = 41
			patch_size_list = [6, 12, 24]
			downsample_kernal_list = [1, 2, 4, 8]

		elif dataset_name == "mimic":
			max_vars_per_plot = 96
			patch_size_list = [12, 24]
			downsample_kernal_list = [1, 2, 4]

		colors = [
			'#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
			'#bcbd22', '#17becf', '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94',
			'#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5', '#393b79', '#5254a3', '#6b6ecf', '#9c9ede',
			'#637939', '#8ca252', '#b5cf6b', '#cedb9c', '#8c6d31', '#bd9e39', '#e7ba52', '#e7cb94',
			'#843c39', '#ad494a', '#d6616b', '#e7969c', '#7b4173', '#a55194', '#ce6dbd', '#de9ed6',
			'#393b79', '#5254a3', '#6b6ecf', '#9c9ede', '#637939', '#8ca252', '#b5cf6b', '#cedb9c',
			'#8c6d31', '#bd9e39', '#e7ba52', '#e7cb94', '#843c39', '#ad494a', '#d6616b', '#e7969c',
			'#7b4173', '#a55194', '#ce6dbd', '#de9ed6', '#3182bd', '#6baed6', '#9ecae1', '#c6dbef',
			'#e6550d', '#fd8d3c', '#fdae6b', '#fdd0a2', '#31a354', '#74c476', '#a1d99b', '#c7e9c0',
			'#756bb1', '#9e9ac8', '#bcbddc', '#dadaeb', '#636363', '#969696', '#bdbdbd', '#d9d9d9',
			'#637939', '#8ca252', '#b5cf6b', '#cedb9c', '#8c6d31', '#bd9e39', '#e7ba52', '#e7cb94',
			'#843c39', '#ad494a', '#d6616b', '#e7969c', '#7b4173', '#a55194', '#ce6dbd', '#de9ed6'
		]

		# for downsample_step in downsample_kernal_list:
		# 	for tup in total_dataset[:50]:
		# 		his_mask = tup[1] < 24
		# 		time_tensor = tup[1][his_mask].cpu()
		# 		value_tensor = tup[2][his_mask].cpu()
		# 		mask_tensor = tup[3][his_mask].cpu()
		# 		T, V = value_tensor.shape
		# 		num_plots = int(np.ceil(V / max_vars_per_plot))
		#
		# 		fig, axes = plt.subplots(num_plots, 1, figsize=(10, 5 * num_plots))
		# 		if num_plots == 1:
		# 			axes = [axes]  # Ensure axes is iterable even with a single plot
		#
		# 		for i in range(num_plots):
		# 			ax = axes[i]
		# 			var_start = i * max_vars_per_plot
		# 			var_end = min((i + 1) * max_vars_per_plot, V)
		#
		# 			for j in range(var_start, var_end):
		# 				mask = mask_tensor[:, j] == 1
		# 				time_valid = time_tensor[mask][::downsample_step]
		# 				values_valid = value_tensor[mask, j][::downsample_step]
		# 				if len(time_valid) > 3:  # Only interpolate if there are at least 2 points
		# 					# spline = interp1d(time_valid, values_valid, kind='cubic')
		# 					spline = make_interp_spline(time_valid, values_valid, k=3)
		# 					time_dense = np.linspace(time_valid.min(), time_valid.max(), 300)
		# 					values_smooth = spline(time_dense)
		# 					ax.plot(time_dense, values_smooth, color=colors[j])
		# 					ax.plot(time_valid, values_valid,  linewidth=0, marker='o', color=colors[j])
		# 				else:
		# 					ax.plot(time_valid, values_valid, marker='o', color=colors[j])
		#
		# 			ax.set_xlabel('Time (h)', fontsize=20)
		# 			ax.set_ylabel('Value', fontsize=20)
		# 			ax.tick_params(axis='x', labelsize=20)
		# 			ax.tick_params(axis='y', labelsize=20)
		# 			# ax.set_title(f'Variables {var_start + 1} to {var_end}')
		# 			# ax.legend()
		# 			# ax.grid(True)
		#
		# 		plt.tight_layout()
		# 		plt.savefig('./vis/' + dataset_name + '/his' + str(tup[0]) + 'Down' + str(downsample_step) + '.svg', bbox_inches='tight')
		# 		plt.show()
		#
		# # 绘制多尺度
		# for patch_size in patch_size_list:
		# 	patch_num = int(48 // patch_size)
		# 	for tup in total_dataset[:50]:
		# 		time_list = []
		# 		value_list = []
		# 		mask_list = []
		# 		his_mask = tup[1] < 24
		#
		# 		for i in range(patch_num):
		# 			a = torch.where(tup[1][his_mask] > i * patch_size)[0]
		# 			b = torch.where(tup[1][his_mask] < (i + 1) * patch_size)[0]
		# 			# cur_ind = [val for val in a if val in b]
		# 			cur_ind = torch.tensor([val for val in a if val in b])
		# 			if cur_ind.shape[0] == 0:
		# 				continue
		# 			obs_num = torch.sum(tup[3][his_mask][cur_ind], dim=0).cpu()
		# 			time_matrix = torch.repeat_interleave(tup[1][his_mask].unsqueeze(-1), max_vars_per_plot, dim=-1)[cur_ind].cpu()
		# 			mask_time_matrix = tup[3][his_mask][cur_ind].cpu() * time_matrix
		# 			time_list.append(torch.nan_to_num(torch.sum(mask_time_matrix, dim=0).cpu() / obs_num, 0))
		# 			value_list.append(torch.nan_to_num(torch.sum(tup[2][his_mask][cur_ind], dim=0).cpu() / obs_num, 0))
		# 			mask_list.append(torch.sum(tup[3][his_mask][cur_ind], dim=0).cpu() > 0)
		#
		# 		time_tensor = torch.concat(time_list).view(-1, max_vars_per_plot)
		# 		value_tensor = torch.concat(value_list).view(-1, max_vars_per_plot)
		# 		mask_tensor = torch.concat(mask_list).view(-1, max_vars_per_plot)
		#
		# 		T, V = value_tensor.shape
		# 		num_plots = int(np.ceil(V / max_vars_per_plot))
		#
		# 		fig, axes = plt.subplots(num_plots, 1, figsize=(10, 5 * num_plots))
		# 		if num_plots == 1:
		# 			axes = [axes]  # Ensure axes is iterable even with a single plot
		#
		# 		for i in range(num_plots):
		# 			ax = axes[i]
		# 			var_start = i * max_vars_per_plot
		# 			var_end = min((i + 1) * max_vars_per_plot, V)
		#
		# 			for j in range(var_start, var_end):
		# 				mask = mask_tensor[:, j] == 1
		# 				time_valid = time_tensor[mask, j]
		# 				values_valid = value_tensor[mask, j]
		# 				if len(time_valid) > 3:  # Only interpolate if there are at least 2 points
		# 					# spline = interp1d(time_valid, values_valid, kind='cubic')
		# 					try:
		# 						spline = make_interp_spline(time_valid, values_valid, k=3)
		# 					except:
		# 						continue
		# 					time_dense = np.linspace(time_valid.min(), time_valid.max(), 300)
		# 					values_smooth = spline(time_dense)
		# 					ax.plot(time_dense, values_smooth, color=colors[j])
		# 					ax.plot(time_valid, values_valid, linewidth=0, marker='o', color=colors[j])
		# 				else:
		# 					ax.plot(time_valid, values_valid, marker='o', color=colors[j])
		#
		# 			ax.set_xlabel('Time (h)', fontsize=20)
		# 			ax.set_ylabel('Value', fontsize=20)
		# 			ax.tick_params(axis='x', labelsize=20)
		# 			ax.tick_params(axis='y', labelsize=20)
		#
		# 		plt.tight_layout()
		# 		plt.savefig('./vis/' + dataset_name + '/his' + str(tup[0]) + '_P' + str(patch_num) + '.svg',
		# 					bbox_inches='tight')
		# 		plt.show()

		# 绘制样本分布图
		# if dataset_name == "physionet":
		# 	x = np.arange(217)
		# 	y = np.zeros(217)
		# elif dataset_name == "mimic":
		# 	x = np.arange(644)
		# 	y = np.zeros(644)
		# time_span = []
		# length = []
		# colors = []
		# min_len = 10000000
		# max_len = 0
		# sum_len = 0
		#
		# for tup in total_dataset:
		# 	cur_len = len(tup[1])
		# 	length.append(cur_len)
		# 	time_span.append(int(tup[1][-1].max()))
		# 	min_len = cur_len if cur_len < min_len else min_len
		# 	max_len = max_len if cur_len < max_len else cur_len
		# 	sum_len += cur_len
		# 	y[cur_len] += 1
		# for tup in total_dataset:
		# 	colors.append(y[len(tup[1])])
		# # y = (y - np.min(y)) / (np.max(y) - np.min(y))
		# y = y / y.sum()
		# # 创建图形和轴
		# fig, ax1 = plt.subplots()
		# # 绘制散点图（左 y 坐标轴）
		# scatter = ax1.scatter(length, time_span, color='blue', label='Sample Point', alpha=0.1)
		# ax1.set_xlabel('Length', fontsize=25)
		# ax1.set_ylabel('Time Span (hours)', color='blue', fontsize=25)
		# ax1.tick_params(axis='x', labelsize=20)
		# ax1.tick_params(axis='y', labelcolor='blue', labelsize=20)
		# # 创建共享 x 轴的第二个 y 轴
		# ax2 = ax1.twinx()
		# # 绘制曲线图（右 y 坐标轴）
		# line, = ax2.plot(x, y, color='red', label='Distribution curve')
		# ax2.set_ylabel('Ratios of samples', color='red', fontsize=25)
		# ax2.tick_params(axis='y', labelcolor='red', labelsize=20)
		# # 添加图例
		# fig.legend(loc='upper right', bbox_to_anchor=(0.85, 0.5), fontsize=20)
		# # 添加标题
		# # plt.title('Distribution of Sample Length and Time Span on ' + dataset_name + ' Dataset')
		# plt.savefig(dataset_name + 'combined.svg', bbox_inches='tight')
		# plt.show()


		# Shuffle and split

		seen_data, test_data = model_selection.train_test_split(total_dataset, train_size= 0.8, random_state = 42, shuffle = True)
		train_data, val_data = model_selection.train_test_split(seen_data, train_size= 0.75, random_state = 42, shuffle = False)
		print("Dataset n_samples:", len(total_dataset), len(train_data), len(val_data), len(test_data))
		test_record_ids = [record_id for record_id, tt, vals, mask in test_data]
		print("Test record ids (first 20):", test_record_ids[:20])
		print("Test record ids (last 20):", test_record_ids[-20:])
		# print(train_data[0][0], test_data[0][0])

		record_id, tt, vals, mask = train_data[0]

		n_samples = len(total_dataset)
		input_dim = vals.size(-1)

		batch_size = min(min(len(seen_data), args.batch_size), args.n)
		if args.model == 'GRUD':
			data_min, data_max, time_max, x_mean = get_data_min_max_mean(seen_data, device)  # (n_dim,), (n_dim,)
			# x_mean = (x_mean - data_min) / (data_max - data_min)
			x_mean = torch.nan_to_num((x_mean - data_min) / (data_max - data_min))
		else:
			data_min, data_max, time_max = get_data_min_max(seen_data, device) # (n_dim,), (n_dim,)
			x_mean = None
		# print(data_min.shape, data_min, data_min.min())
		# print(data_max.shape, data_max, data_max.min())

		if(patch_ts):
			collate_fn = patch_variable_time_collate_fn
		elif(args.model == "CRU"):
			collate_fn = variable_time_collate_fn_CRU
		elif(args.model == "Latent_ODE"):
			collate_fn = variable_time_collate_fn_ODE
		elif (args.model == "Warpformer" or args.model == "GRUD"):
			collate_fn = variable_time_collate_fn_warpformer
		elif (args.model == 'GraFITi' or args.model == 'mTAND' or args.model == 'NeuralFlow'):
			collate_fn = variable_time_collate_fn_grafiti
		else:
			collate_fn = variable_time_collate_fn



		train_dataloader = DataLoader(train_data, batch_size= batch_size, shuffle=True, 
			collate_fn= lambda batch: collate_fn(batch, args, device, data_type = "train",
				data_min = data_min, data_max = data_max, time_max = time_max))
		val_dataloader = DataLoader(val_data, batch_size= batch_size, shuffle=False, 
			collate_fn= lambda batch: collate_fn(batch, args, device, data_type = "val",
				data_min = data_min, data_max = data_max, time_max = time_max))
		test_dataloader = DataLoader(test_data, batch_size = batch_size, shuffle=False, 
			collate_fn= lambda batch: collate_fn(batch, args, device, data_type = "test",
				data_min = data_min, data_max = data_max, time_max = time_max))

		data_objects = {
					"train_dataloader": utils.inf_generator(train_dataloader), 
					"val_dataloader": utils.inf_generator(val_dataloader),
					"test_dataloader": utils.inf_generator(test_dataloader),
					"input_dim": input_dim,
					"n_train_batches": len(train_dataloader),
					"n_val_batches": len(val_dataloader),
					"n_test_batches": len(test_dataloader),
					# "attr": total_dataset.params, #optional
					"data_max": data_max, #optional
					"data_min": data_min,
					"time_max": time_max,
					'x_mean': x_mean
					} #optional

		if(length_stat):
			max_input_len, max_pred_len, median_len = get_seq_length(args, total_dataset)
			data_objects["max_input_len"] = max_input_len.item()
			data_objects["max_pred_len"] = max_pred_len.item()
			data_objects["median_len"] = median_len.item()
			print(data_objects["max_input_len"], data_objects["max_pred_len"], data_objects["median_len"])

		return data_objects

	##################################################################
	### USHCN dataset ###
	elif dataset_name == "ushcn":
		args.n_months = 48 # 48 monthes
		# args.pred_window = 1 # predict future one month

		### list of tuples (record_id, tt, vals, mask) ###
		total_dataset = USHCN('../data/ushcn/', n_samples = args.n, device = device)

		max_vars_per_plot = 5
		colors = [
			'#1f77b4',  # 蓝色
			'#ff7f0e',  # 橙色
			'#2ca02c',  # 绿色
			'#d62728',  # 红色
			'#9467bd',  # 紫色
			'#8c564b',  # 棕色
			'#e377c2',  # 粉色
			'#7f7f7f',  # 灰色
			'#bcbd22',  # 黄绿色
			'#17becf',  # 青色
			'#ffbb78',  # 浅橙色
			'#ff9896'  # 浅红色
		]
		# downsample_kernal_list = [1, 2, 4, 8, 16, 32]
		# for downsample_step in downsample_kernal_list:
		#
		# 	for tup in total_dataset[:1]:
		# 		his_mask = tup[1] < 24
		# 		time_tensor = tup[1][his_mask].cpu()
		# 		value_tensor = tup[2][his_mask].cpu()
		# 		mask_tensor = tup[3][his_mask].cpu()
		# 		T, V = value_tensor.shape
		# 		num_plots = int(np.ceil(V / max_vars_per_plot))
		#
		# 		fig, axes = plt.subplots(num_plots, 1, figsize=(10, 5 * num_plots))
		# 		if num_plots == 1:
		# 			axes = [axes]  # Ensure axes is iterable even with a single plot
		#
		# 		for i in range(num_plots):
		# 			ax = axes[i]
		# 			var_start = i * max_vars_per_plot
		# 			var_end = min((i + 1) * max_vars_per_plot, V)
		#
		# 			for j in range(var_start, var_end):
		# 				mask = mask_tensor[:, j] == 1
		#
		# 				time_valid = time_tensor[mask][::downsample_step]
		# 				values_valid = value_tensor[mask, j][::downsample_step]
		# 				if len(time_valid) > 3:  # Only interpolate if there are at least 2 points
		# 					# spline = interp1d(time_valid, values_valid, kind='cubic')
		# 					spline = make_interp_spline(time_valid, values_valid, k=3)
		# 					time_dense = np.linspace(time_valid.min(), time_valid.max(), 300)
		# 					values_smooth = spline(time_dense)
		# 					ax.plot(time_dense, values_smooth, color=colors[j])
		# 					ax.plot(time_valid, values_valid,  linewidth=0, marker='o', color=colors[j])
		# 				else:
		# 					ax.plot(time_valid, values_valid, marker='o', color=colors[j])
		# 				# ax.plot(time_tensor[mask], value_tensor[mask, j], marker='o')
		#
		# 			ax.set_xlabel('Time (month)', fontsize=20)
		# 			ax.set_ylabel('Value', fontsize=20)
		# 			ax.tick_params(axis='x', labelsize=20)
		# 			ax.tick_params(axis='y', labelsize=20)
		# 			# ax.set_title(f'Variables {var_start + 1} to {var_end}')
		# 			# ax.legend()
		# 			# ax.grid(True)
		#
		# 		plt.tight_layout()
		# 		plt.savefig('./vis/' + dataset_name + '/his' + str(tup[0]) + 'Down' + str(downsample_step) + '.svg', bbox_inches='tight')
		# 		plt.show()


		# 绘制多尺度
		# patch_size_list = [1.5, 3, 6, 12, 24]
		# for patch_size in patch_size_list:
		# 	patch_num = int(48 // patch_size)
		# 	for tup in total_dataset[:1]:
		# 		his_mask = tup[1] < 24
		# 		if len(his_mask) == 0:
		# 			continue
		# 		time_list = []
		# 		value_list = []
		# 		mask_list = []
		# 		for i in range(patch_num):
		# 			# cur_ind = (torch.where(tup[1] > i * patch_size) and torch.where(tup[1] < (i + 1) * patch_size))[0]
		# 			a = torch.where(tup[1][his_mask] > i * patch_size)[0]
		# 			b = torch.where(tup[1][his_mask] < (i + 1) * patch_size)[0]
		# 			# cur_ind = [val for val in a if val in b]
		# 			cur_ind = torch.tensor([val for val in a if val in b])
		# 			if cur_ind.shape[0] == 0:
		# 				continue
		# 			obs_num = torch.sum(tup[3][his_mask][cur_ind], dim=0).cpu()
		# 			time_matrix = torch.repeat_interleave(tup[1][his_mask].unsqueeze(-1), 5, dim=-1)[cur_ind].cpu()
		# 			mask_time_matrix = tup[3][his_mask][cur_ind].cpu() * time_matrix
		# 			time_list.append(torch.nan_to_num(torch.sum(mask_time_matrix, dim=0).cpu() / obs_num, 0))
		# 			value_list.append(torch.nan_to_num(torch.sum(tup[2][his_mask][cur_ind], dim=0).cpu() / obs_num, 0))
		# 			mask_list.append(torch.sum(tup[3][his_mask][cur_ind], dim=0).cpu() > 0)
		# 		try:
		# 			time_tensor = torch.concat(time_list).view(-1, 5)
		# 			value_tensor = torch.concat(value_list).view(-1, 5)
		# 			mask_tensor = torch.concat(mask_list).view(-1, 5)
		# 		except:
		# 			continue
		# 		T, V = value_tensor.shape
		# 		num_plots = int(np.ceil(V / max_vars_per_plot))
		#
		# 		fig, axes = plt.subplots(num_plots, 1, figsize=(10, 5 * num_plots))
		# 		if num_plots == 1:
		# 			axes = [axes]  # Ensure axes is iterable even with a single plot
		#
		# 		for i in range(num_plots):
		# 			ax = axes[i]
		# 			var_start = i * max_vars_per_plot
		# 			var_end = min((i + 1) * max_vars_per_plot, V)
		#
		# 			for j in range(var_start, var_end):
		# 				mask = mask_tensor[:, j] == 1
		# 				time_valid = time_tensor[mask, j]
		# 				values_valid = value_tensor[mask, j]
		# 				if len(time_valid) > 3:  # Only interpolate if there are at least 2 points
		# 					# spline = interp1d(time_valid, values_valid, kind='cubic')
		# 					try:
		# 						spline = make_interp_spline(time_valid, values_valid, k=3)
		# 					except:
		# 						continue
		# 					time_dense = np.linspace(time_valid.min(), time_valid.max(), 300)
		# 					values_smooth = spline(time_dense)
		# 					ax.plot(time_dense, values_smooth, color=colors[j])
		# 					ax.plot(time_valid, values_valid, linewidth=0, marker='o', color=colors[j])
		# 				else:
		# 					ax.plot(time_valid, values_valid, marker='o', color=colors[j])
		#
		# 			ax.set_xlabel('Time (month)', fontsize=20)
		# 			ax.set_ylabel('Value', fontsize=20)
		# 			ax.tick_params(axis='x', labelsize=20)
		# 			ax.tick_params(axis='y', labelsize=20)
		#
		# 		plt.tight_layout()
		# 		plt.savefig('./vis/' + dataset_name + '/his' + str(tup[0]) + '_P' + str(patch_num) + '.svg',
		# 					bbox_inches='tight')
		# 		plt.show()

		# x = np.arange(371)
		# y = np.zeros(371)
		# time_span = []
		# length = []
		# colors = []
		# min_len = 10000000
		# max_len = 0
		# sum_len = 0
		#
		# for tup in total_dataset:
		# 	cur_len = len(tup[1])
		# 	length.append(cur_len)
		# 	time_span.append(int(tup[1][-1].max()))
		# 	min_len = cur_len if cur_len < min_len else min_len
		# 	max_len = max_len if cur_len < max_len else cur_len
		# 	sum_len += cur_len
		# 	y[cur_len] += 1
		# for tup in total_dataset:
		# 	colors.append(y[len(tup[1])])
		# y = y / y.sum()
		# # 创建图形和轴
		# fig, ax1 = plt.subplots()
		# # 绘制散点图（左 y 坐标轴）
		# scatter = ax1.scatter(length, time_span, color='blue', label='Sample Point', alpha=0.1)
		# ax1.set_xlabel('Length', fontsize=25)
		# ax1.set_ylabel('Time Span (hours)', color='blue', fontsize=25)
		# ax1.tick_params(axis='x', labelsize=20)
		# ax1.tick_params(axis='y', labelcolor='blue', labelsize=20)
		# # 创建共享 x 轴的第二个 y 轴
		# ax2 = ax1.twinx()
		# # 绘制曲线图（右 y 坐标轴）
		# line, = ax2.plot(x, y, color='red', label='Distribution curve')
		# ax2.set_ylabel('Ratios of samples', color='red', fontsize=25)
		# ax2.tick_params(axis='y', labelcolor='red', labelsize=20)
		# # 添加图例
		# fig.legend(loc='lower left', bbox_to_anchor=(0.15, 0.25), fontsize=20)
		# # fig.legend(fontsize=20)
		# # 添加标题
		# # plt.title('Distribution of Sample Length and Time Span on ' + dataset_name + ' Dataset')
		# plt.savefig(dataset_name + 'combined.svg', bbox_inches='tight')



		# Shuffle and split
		seen_data, test_data = model_selection.train_test_split(total_dataset, train_size= 0.8, random_state = 42, shuffle = True)
		train_data, val_data = model_selection.train_test_split(seen_data, train_size= 0.75, random_state = 42, shuffle = False)
		print("Dataset n_samples:", len(total_dataset), len(train_data), len(val_data), len(test_data))
		test_record_ids = [record_id for record_id, tt, vals, mask in test_data]
		print("Test record ids (first 20):", test_record_ids[:20])
		print("Test record ids (last 20):", test_record_ids[-20:])
		# print(train_data[0][0], test_data[0][0])

		record_id, tt, vals, mask = train_data[0]

		n_samples = len(total_dataset)
		input_dim = vals.size(-1)
		if args.model == 'GRUD':
			data_min, data_max, time_max, x_mean = get_data_min_max_mean(seen_data, device)  # (n_dim,), (n_dim,)
		else:
			data_min, data_max, time_max = get_data_min_max(seen_data, device)  # (n_dim,), (n_dim,)
			x_mean = None
		# print(data_min.shape, data_min, data_min.min())
		# print(data_max.shape, data_max, data_max.min())

		if(patch_ts):
			collate_fn = USHCN_patch_variable_time_collate_fn
		elif(args.model == "CRU"):
			collate_fn = USHCN_variable_time_collate_fn_CRU
		elif(args.model == "Latent_ODE"):
			collate_fn = USHCN_variable_time_collate_fn_ODE
		elif (args.model == "Warpformer" or args.model == 'GRUD'):
			collate_fn = USHCN_variable_time_collate_fn_warpformer
		elif (args.model == 'GraFITi' or args.model == 'mTAND' or args.model == 'NeuralFlow'):
			collate_fn = USHCN_variable_time_collate_fn_grafiti
		else:
			collate_fn = USHCN_variable_time_collate_fn


		train_data = USHCN_time_chunk(train_data, args, device)
		val_data = USHCN_time_chunk(val_data, args, device)
		test_data = USHCN_time_chunk(test_data, args, device)
		batch_size = args.batch_size
		print("Dataset n_samples after time split:", len(train_data)+len(val_data)+len(test_data),\
			len(train_data), len(val_data), len(test_data))
		train_dataloader = DataLoader(train_data, batch_size= batch_size, shuffle=True, 
			collate_fn= lambda batch: collate_fn(batch, args, device, time_max = time_max, data_max=data_max, data_min=data_min))
		val_dataloader = DataLoader(val_data, batch_size= batch_size, shuffle=False, 
			collate_fn= lambda batch: collate_fn(batch, args, device, time_max = time_max, data_max=data_max, data_min=data_min))
		test_dataloader = DataLoader(test_data, batch_size = batch_size, shuffle=False, 
			collate_fn= lambda batch: collate_fn(batch, args, device, time_max = time_max, data_max=data_max, data_min=data_min))

		data_objects = {
					"train_dataloader": utils.inf_generator(train_dataloader), 
					"val_dataloader": utils.inf_generator(val_dataloader),
					"test_dataloader": utils.inf_generator(test_dataloader),
					"input_dim": input_dim,
					"n_train_batches": len(train_dataloader),
					"n_val_batches": len(val_dataloader),
					"n_test_batches": len(test_dataloader),
					# "attr": total_dataset.params, #optional
					"data_max": data_max, #optional
					"data_min": data_min,
					"time_max": time_max,
					'x_mean': x_mean
		} #optional

		if(length_stat):
			max_input_len, max_pred_len, median_len = USHCN_get_seq_length(args, train_data+val_data+test_data)
			data_objects["max_input_len"] = max_input_len.item()
			data_objects["max_pred_len"] = max_pred_len.item()
			data_objects["median_len"] = median_len.item()
			# data_objects["batch_size"] = args.batch_size * (args.n_months - args.pred_window + 1 - args.history)
			print(data_objects["max_input_len"], data_objects["max_pred_len"], data_objects["median_len"])

		return data_objects
		

	##################################################################
	### Activity dataset ###
	elif dataset_name == "activity":
		args.pred_window = 4000 - args.history # predict future 1000 ms
		# args.pred_window = 1000 #

		total_dataset = PersonActivity('../data/activity/', n_samples = args.n, download=True, device = device)



		# Shuffle and split
		seen_data, test_data = model_selection.train_test_split(total_dataset, train_size= 0.8, random_state = 42, shuffle = True)
		train_data, val_data = model_selection.train_test_split(seen_data, train_size= 0.75, random_state = 42, shuffle = False)
		print("Dataset n_samples:", len(total_dataset), len(train_data), len(val_data), len(test_data))
		test_record_ids = [record_id for record_id, tt, vals, mask in test_data]
		print("Test record ids (first 20):", test_record_ids[:20])
		print("Test record ids (last 20):", test_record_ids[-20:])
		# print(train_data[0][0], test_data[0][0])

		record_id, tt, vals, mask = train_data[0]

		n_samples = len(total_dataset)
		input_dim = vals.size(-1)

		batch_size = min(min(len(seen_data), args.batch_size), args.n)

		if args.model == 'GRUD':
			data_min, data_max, _, x_mean = get_data_min_max_mean(seen_data, device)  # (n_dim,), (n_dim,)
			x_mean = (x_mean - data_min) / (data_max - data_min)
		else:
			data_min, data_max, _ = get_data_min_max(seen_data, device)  # (n_dim,), (n_dim,)
			x_mean = None

		time_max = torch.tensor(args.history + args.pred_window)
		print('manual set time_max:', time_max)

		if(patch_ts):
			collate_fn = patch_variable_time_collate_fn
		elif(args.model == "CRU"):
			collate_fn = variable_time_collate_fn_CRU
		elif(args.model == "Latent_ODE"):
			collate_fn = variable_time_collate_fn_ODE
		elif (args.model == 'GraFITi' or args.model == 'mTAND'):
			collate_fn = variable_time_collate_fn_grafiti
		elif (args.model == 'Warpformer' or args.model == 'GRUD'):
			collate_fn = variable_time_collate_fn_warpformer
		elif (args.model == 'NeuralFlow'):
			collate_fn = variable_time_collate_fn_neuralflow
		else:
			collate_fn = variable_time_collate_fn

		train_data = Activity_time_chunk(train_data, args, device)
		val_data = Activity_time_chunk(val_data, args, device)
		test_data = Activity_time_chunk(test_data, args, device)
		batch_size = args.batch_size

		max_vars_per_plot = 12

		colors = [
			'#1f77b4',  # 蓝色
			'#ff7f0e',  # 橙色
			'#2ca02c',  # 绿色
			'#d62728',  # 红色
			'#9467bd',  # 紫色
			'#8c564b',  # 棕色
			'#e377c2',  # 粉色
			'#7f7f7f',  # 灰色
			'#bcbd22',  # 黄绿色
			'#17becf',  # 青色
			'#ffbb78',  # 浅橙色
			'#ff9896'  # 浅红色
		]

		# downsample_kernal_list = [1, 2, 4, 8]
		# for downsample_step in downsample_kernal_list:
		# # 绘制原始尺度
		# 	for tup in (train_data + val_data + test_data)[:1]:
		# 		his_mask = tup[1] < 3000
		# 		time_tensor = tup[1][his_mask].cpu()
		# 		value_tensor = tup[2][his_mask].cpu()
		# 		mask_tensor = tup[3][his_mask].cpu()
		# 		T, V = value_tensor.shape
		# 		num_plots = int(np.ceil(V / max_vars_per_plot))
		#
		# 		fig, axes = plt.subplots(num_plots, 1, figsize=(10, 5 * num_plots))
		# 		if num_plots == 1:
		# 			axes = [axes]  # Ensure axes is iterable even with a single plot
		#
		# 		for i in range(num_plots):
		# 			ax = axes[i]
		# 			var_start = i * max_vars_per_plot
		# 			var_end = min((i + 1) * max_vars_per_plot, V)
		#
		# 			for j in range(var_start, var_end):
		# 				mask = mask_tensor[:, j] == 1
		#
		# 				time_valid = time_tensor[mask][::downsample_step]
		# 				values_valid = value_tensor[mask, j][::downsample_step]
		# 				if len(time_valid) > 3:  # Only interpolate if there are at least 2 points
		# 					# spline = interp1d(time_valid, values_valid, kind='cubic')
		# 					spline = make_interp_spline(time_valid, values_valid, k=3)
		# 					time_dense = np.linspace(time_valid.min(), time_valid.max(), 300)
		# 					values_smooth = spline(time_dense)
		# 					ax.plot(time_dense, values_smooth, color=colors[j])
		# 					ax.plot(time_valid, values_valid,  linewidth=0, marker='o', color=colors[j])
		# 				else:
		# 					ax.plot(time_valid, values_valid, marker='o', color=colors[j])
		# 				# ax.plot(time_tensor[mask], value_tensor[mask, j], marker='o')
		#
		# 			ax.set_xlabel('Time (ms)', fontsize=20)
		# 			ax.set_ylabel('Value', fontsize=20)
		# 			ax.tick_params(axis='x', labelsize=20)
		# 			ax.tick_params(axis='y', labelsize=20)
		#
		# 		# ax.set_title(f'Variables {var_start + 1} to {var_end}')
		# 			# ax.legend()
		# 			# ax.grid(True)
		#
		# 		plt.tight_layout()
		# 		plt.savefig('./vis/' + dataset_name + '/his' + str(tup[0]) + 'Down' + str(downsample_step) + '.svg', bbox_inches='tight')
		# 		plt.show()
		#
		# # 绘制多尺度
		# patch_size_list = [375, 750, 1500]
		# for patch_size in patch_size_list:
		# 	patch_num = 4000 // patch_size
		# 	for tup in (train_data + val_data + test_data)[:1]:
		# 		his_mask = tup[1] < 3000
		# 		time_list = []
		# 		value_list = []
		# 		mask_list = []
		# 		for i in range(patch_num):
		# 			a = torch.where(tup[1][his_mask] > i * patch_size)[0]
		# 			b = torch.where(tup[1][his_mask] < (i + 1) * patch_size)[0]
		# 			cur_ind = torch.tensor([val for val in a if val in b])
		# 			if cur_ind.shape[0] == 0:
		# 				continue
		# 			obs_num = torch.sum(tup[3][his_mask][cur_ind], dim=0).cpu()
		# 			time_matrix = torch.repeat_interleave(tup[1][his_mask].unsqueeze(-1), 12, dim=-1)[cur_ind].cpu()
		# 			mask_time_matrix = tup[3][his_mask][cur_ind].cpu() * time_matrix
		# 			time_list.append(torch.nan_to_num(torch.sum(mask_time_matrix, dim=0).cpu() / obs_num, 0))
		# 			value_list.append(torch.nan_to_num(torch.sum(tup[2][his_mask][cur_ind], dim=0).cpu() / obs_num, 0))
		# 			mask_list.append(torch.sum(tup[3][his_mask][cur_ind], dim=0).cpu() > 0)
		#
		# 			# time_sum = torch.sum(tup[1][cur_ind]).view(-1)
		# 			# time_list.append(torch.mean(tup[1][cur_ind]).view(-1).cpu())
		# 			# value_list.append(torch.mean(tup[2][cur_ind], dim=0).cpu())
		# 			# mask_list.append(torch.sum(tup[3][cur_ind], dim=0).cpu() > 0)
		#
		# 		time_tensor = torch.concat(time_list).view(-1, input_dim)
		# 		value_tensor = torch.concat(value_list).view(-1, input_dim)
		# 		mask_tensor = torch.concat(mask_list).view(-1, input_dim)
		#
		# 		T, V = value_tensor.shape
		# 		num_plots = int(np.ceil(V / max_vars_per_plot))
		#
		# 		fig, axes = plt.subplots(num_plots, 1, figsize=(10, 5 * num_plots))
		# 		if num_plots == 1:
		# 			axes = [axes]  # Ensure axes is iterable even with a single plot
		#
		# 		for i in range(num_plots):
		# 			ax = axes[i]
		# 			var_start = i * max_vars_per_plot
		# 			var_end = min((i + 1) * max_vars_per_plot, V)
		#
		# 			for j in range(var_start, var_end):
		# 				mask = mask_tensor[:, j] == 1
		# 				time_valid = time_tensor[mask, j]
		# 				values_valid = value_tensor[mask, j]
		# 				if len(time_valid) > 3:  # Only interpolate if there are at least 2 points
		# 					# spline = interp1d(time_valid, values_valid, kind='cubic')
		# 					spline = make_interp_spline(time_valid, values_valid, k=3)
		# 					time_dense = np.linspace(time_valid.min(), time_valid.max(), 300)
		# 					values_smooth = spline(time_dense)
		# 					ax.plot(time_dense, values_smooth, color=colors[j])
		# 					ax.plot(time_valid, values_valid,  linewidth=0, marker='o', color=colors[j])
		# 				else:
		# 					ax.plot(time_valid, values_valid, marker='o', color=colors[j])
		#
		# 			ax.set_xlabel('Time (ms)', fontsize=20)
		# 			ax.set_ylabel('Value', fontsize=20)
		# 			ax.tick_params(axis='x', labelsize=20)
		# 			ax.tick_params(axis='y', labelsize=20)
		# 		# ax.set_title(f'Variables {var_start + 1} to {var_end}')
		# 		# ax.legend()
		# 		# ax.grid(True)
		#
		# 		plt.tight_layout()
		# 		plt.savefig('./vis/' + dataset_name + '/his' + tup[0] + '_P' + str(patch_num) + '.svg', bbox_inches='tight')
		# 		plt.show()

		x = np.arange(131)
		y = np.zeros(131)
		time_span = []
		length = []
		colors = []
		min_len = 10000000
		max_len = 0
		sum_len = 0

		for tup in train_data + val_data + test_data:
			cur_len = len(tup[1])
			length.append(cur_len)
			time_span.append(int(tup[1][-1].max()))
			min_len = cur_len if cur_len < min_len else min_len
			max_len = max_len if cur_len < max_len else cur_len
			sum_len += cur_len
			y[cur_len] += 1
		for tup in train_data + val_data + test_data:
			colors.append(y[len(tup[1])])
		y = y / y.sum()
		# 创建图形和轴
		fig, ax1 = plt.subplots()
		# 绘制散点图（左 y 坐标轴）
		scatter = ax1.scatter(length, time_span, color='blue', label='Sample Point', alpha=0.1)
		ax1.set_xlabel('Length', fontsize=25)
		ax1.set_ylabel('Time Span (ms)', color='blue', fontsize=25)
		ax1.tick_params(axis='x', labelsize=20)
		ax1.tick_params(axis='y', labelcolor='blue', labelsize=20)
		# 创建共享 x 轴的第二个 y 轴
		ax2 = ax1.twinx()
		# 绘制曲线图（右 y 坐标轴）
		line, = ax2.plot(x, y, color='red', label='Distribution curve')
		ax2.set_ylabel('Ratios of samples', color='red', fontsize=25)
		ax2.tick_params(axis='y', labelcolor='red', labelsize=20)
		# 添加图例
		fig.legend(loc='lower left', bbox_to_anchor=(0.15, 0.5), fontsize=22)
		# 添加标题
		# plt.title('Distribution of Sample Length and Time Span on ' + dataset_name + ' Dataset')
		plt.savefig(dataset_name + 'combined.svg', bbox_inches='tight')

		print("Dataset n_samples after time split:", len(train_data)+len(val_data)+len(test_data),\
			len(train_data), len(val_data), len(test_data))
		train_dataloader = DataLoader(train_data, batch_size= batch_size, shuffle=True, 
			collate_fn= lambda batch: collate_fn(batch, args, device, data_type = "train",
				data_min = data_min, data_max = data_max, time_max = time_max))
		val_dataloader = DataLoader(val_data, batch_size= batch_size, shuffle=False, 
			collate_fn= lambda batch: collate_fn(batch, args, device, data_type = "val",
				data_min = data_min, data_max = data_max, time_max = time_max))
		test_dataloader = DataLoader(test_data, batch_size = batch_size, shuffle=False, 
			collate_fn= lambda batch: collate_fn(batch, args, device, data_type = "test",
				data_min = data_min, data_max = data_max, time_max = time_max))

		data_objects = {
					"train_dataloader": utils.inf_generator(train_dataloader), 
					"val_dataloader": utils.inf_generator(val_dataloader),
					"test_dataloader": utils.inf_generator(test_dataloader),
					"input_dim": input_dim,
					"n_train_batches": len(train_dataloader),
					"n_val_batches": len(val_dataloader),
					"n_test_batches": len(test_dataloader),
					# "attr": total_dataset.params, #optional
					"data_max": data_max, #optional
					"data_min": data_min,
					"time_max": time_max,
					'x_mean': x_mean
					} #optional

		if(length_stat):
			max_input_len, max_pred_len, median_len = Activity_get_seq_length(args, train_data+val_data+test_data)
			data_objects["max_input_len"] = max_input_len.item()
			data_objects["max_pred_len"] = max_pred_len.item()
			data_objects["median_len"] = median_len.item()
			print(data_objects["max_input_len"], data_objects["max_pred_len"], data_objects["median_len"])

		return data_objects
	
