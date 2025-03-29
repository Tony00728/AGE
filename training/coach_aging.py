import os
import random
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import gc

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from utils import common, train_utils
from criteria import id_loss, w_norm,shape_loss
from configs import data_configs
from datasets.images_dataset import ImagesDataset
from datasets.augmentations import AgeTransformer
from criteria.lpips.lpips import LPIPS
from criteria.aging_loss import AgingLoss
from models.psp import pSp
from training.ranger import Ranger

#新的
from models.discriminator.id_d import ConditionalDiscriminator, Conditionaldiscriminator_loss
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

from hair import *
from data_loader import CelebASegmentation2

download_models()
resolution = 256
assert torch.cuda.is_available()
torch.backends.cudnn.benchmark = True
model_fname = 'deeplab_model/deeplab_model.pth'
dataset_root = '/home/tony/img'
assert os.path.isdir(dataset_root)
dataset = CelebASegmentation2(dataset_root, crop_size=256)

model = getattr(deeplab, 'resnet101')(
	pretrained=True,
	num_classes=len(dataset.CLASSES),
	num_groups=32,
	weight_std=True,
	beta=False)

model = model.cuda()
model.eval()
checkpoint = torch.load(model_fname)
state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k}
model.load_state_dict(state_dict)


class Coach:
	def __init__(self, opts):
		self.opts = opts

		self.global_step = 0

		self.device = 'cuda'
		self.opts.device = self.device

		# Initialize network
		self.net = pSp(self.opts).to(self.device)  # 在models psp.py

		# Initialize discriminator
		self.discriminator = ConditionalDiscriminator().to(self.device)  #新的
		self.d_lr = self.opts.d_learning_rate
		self.gan_loss_weight = self.opts.gan_loss_weight
		self.d_optim = torch.optim.Adam(self.discriminator.parameters(), self.d_lr, betas=(0.5, 0.99))  #默認betas=(0.9 ,0.999）



		# Initialize loss
		self.mse_loss = nn.MSELoss().to(self.device).eval()
		if self.opts.lpips_lambda > 0:
			self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
		if self.opts.id_lambda > 0:
			self.id_loss = id_loss.IDLoss().to(self.device).eval()

		if self.opts.shape_lambda > 0:
			self.landmark_detector = shape_loss.FaceLandmarkDetector(opts.face_landmarks_path)


		if self.opts.w_norm_lambda > 0:
			self.w_norm_loss = w_norm.WNormLoss(opts=self.opts)
		if self.opts.aging_lambda > 0:
			self.aging_loss = AgingLoss(self.opts)

		# Initialize optimizer
		self.optimizer = self.configure_optimizers()

		# Initialize dataset
		self.train_dataset, self.test_dataset = self.configure_datasets()
		self.train_dataloader = DataLoader(self.train_dataset,
										   batch_size=self.opts.batch_size,
										   shuffle=True,
										   num_workers=int(self.opts.workers),
										   drop_last=True)
		self.test_dataloader = DataLoader(self.test_dataset,
										  batch_size=self.opts.test_batch_size,
										  shuffle=False,
										  num_workers=int(self.opts.test_workers),
										  drop_last=True)

		self.age_transformer = AgeTransformer(target_age=self.opts.target_age)

		# Initialize logger
		log_dir = os.path.join(opts.exp_dir, 'logs')
		os.makedirs(log_dir, exist_ok=True)
		self.logger = SummaryWriter(log_dir=log_dir)

		# Initialize checkpoint dir
		self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
		os.makedirs(self.checkpoint_dir, exist_ok=True)
		self.best_val_loss = None
		if self.opts.save_interval is None:
			self.opts.save_interval = self.opts.max_steps

	def perform_forward_pass(self, x):
		y_hat, latent = self.net.forward(x, return_latents=True)
		return y_hat, latent

	def __set_target_to_source(self, x, input_ages):   #將目標年齡信息添加到輸入影像中
		return [torch.cat((img, age * torch.ones((1, img.shape[1], img.shape[2])).to(self.device)))
				for img, age in zip(x, input_ages)]

	def train(self):
		self.net.train()  #這個net是model裡的psp.py ,訓練psp
		while self.global_step < self.opts.max_steps:   #max_step在train_option.py現在是500000
			for batch_idx, batch in enumerate(self.train_dataloader):

				x, y = batch  #從當前batch中提取輸入 x 和目標 y
				x, y = x.to(self.device).float(), y.to(self.device).float()
				self.optimizer.zero_grad()

				input_ages = self.aging_loss.extract_ages(x) / 100.

				# perform no aging in 33% of the time
				no_aging = random.random() <= (1. / 3)   #隨機生成一個布林值，表示是否不對圖像執行年齡轉換。有 33% 的機率不進行年齡轉換
				if no_aging:
					x_input = self.__set_target_to_source(x=x, input_ages=input_ages)  #不進行年齡轉換
				else:
					x_input = [self.age_transformer(img.cpu()).to(self.device) for img in x]

				x_input = torch.stack(x_input)  #堆疊 x_input 中的所有轉換後圖像，以便進行後續處理
				target_ages = x_input[:, -1, 0, 0] #從 x_input 中提取目標年齡，通常存儲在圖像的最後一個通道中

				# perform forward/backward pass on real images
				y_hat, latent = self.perform_forward_pass(x_input)
				loss, loss_dict, id_logs = self.calc_loss(x, y, y_hat, latent,   # calc_loss 的def在底下
														  target_ages=target_ages,
														  input_ages=input_ages,
														  no_aging=no_aging,
														  data_type="real")
				loss.backward()


				# 循環回去的
				# perform cycle on generate images by setting the target ages to the original input ages
				y_hat_clone = y_hat.clone().detach().requires_grad_(True)
				input_ages_clone = input_ages.clone().detach().requires_grad_(True)
				y_hat_inverse = self.__set_target_to_source(x=y_hat_clone, input_ages=input_ages_clone)
				y_hat_inverse = torch.stack(y_hat_inverse)
				reverse_target_ages = y_hat_inverse[:, -1, 0, 0]
				y_recovered, latent_cycle = self.perform_forward_pass(y_hat_inverse)
				loss, cycle_loss_dict, cycle_id_logs = self.calc_loss(x, y, y_recovered, latent_cycle,
																	  target_ages=reverse_target_ages,
																	  input_ages=input_ages,
																	  no_aging=no_aging,
																	  data_type="cycle")
				loss.backward()
				self.optimizer.step()

				#
				#在每個訓練步驟中，先對生成器執行計算損失和反向傳播後更新生成器權重，接著對鑑別器執行計算損失和反向傳播後更新鑑別器權重。這樣的分開訓練過程有助於穩定和有效地訓練 GAN 模型。
				#身份鑑別器的訓練，目標是原年齡的生成圖片 跟 輸入原圖片 丟到鑑別器中
				self.d_optim.zero_grad()
				#y_recovered, _ = self.perform_forward_pass(y_hat_inverse)  # y_hat_inverse 是生成器產生的圖片

				self.discriminatorloss = Conditionaldiscriminator_loss(self.discriminator,x,y_recovered)

				loss_dict['discriminatorloss'] = self.discriminatorloss.item()

				self.discriminatorloss.backward()  # 鑑別器損失的反向傳播
				self.d_optim.step()  # 更新鑑別器權重

				loss += self.discriminatorloss * self.gan_loss_weight

				#
				# combine the logs of both forwards
				for idx, cycle_log in enumerate(cycle_id_logs):
					id_logs[idx].update(cycle_log) #更新正向傳播和逆向傳播的身份損失log檔
				loss_dict.update(cycle_loss_dict)  #更新損失字典
				loss_dict["loss"] = loss_dict["loss_real"] + loss_dict["loss_cycle"]  #將逆向傳播的損失信息合併到正向傳播的損失信息中

				# Logging related
				if self.global_step % self.opts.image_interval == 0 or \
						(self.global_step < 1000 and self.global_step % 25 == 0):
					self.parse_and_log_images(id_logs, x, y, y_hat, y_recovered,
											  title='images/train/faces')

				if self.global_step % self.opts.board_interval == 0:   #board_interval設50
					self.print_metrics(loss_dict, prefix='train')
					self.log_metrics(loss_dict, prefix='train')

				# Validation related
				val_loss_dict = None
				if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
					val_loss_dict = self.validate()
					if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):
						self.best_val_loss = val_loss_dict['loss']
						self.checkpoint_me(val_loss_dict, is_best=True)

				if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
					if val_loss_dict is not None:
						self.checkpoint_me(val_loss_dict, is_best=False)
					else:
						self.checkpoint_me(loss_dict, is_best=False)

				if self.global_step == self.opts.max_steps:
					print('OMG, finished training!')
					break

				self.global_step += 1





	def validate(self):
		self.net.eval()
		agg_loss_dict = []
		for batch_idx, batch in enumerate(self.test_dataloader):
			x, y = batch
			with torch.no_grad():
				x, y = x.to(self.device).float(), y.to(self.device).float()

				input_ages = self.aging_loss.extract_ages(x) / 100.

				# perform no aging in 33% of the time
				no_aging = random.random() <= (1. / 3)
				if no_aging:
					x_input = self.__set_target_to_source(x=x, input_ages=input_ages)
				else:
					x_input = [self.age_transformer(img.cpu()).to(self.device) for img in x]

				x_input = torch.stack(x_input)
				target_ages = x_input[:, -1, 0, 0]

				# perform forward/backward pass on real images
				y_hat, latent = self.perform_forward_pass(x_input)
				_, cur_loss_dict, id_logs = self.calc_loss(x, y, y_hat, latent,
														   target_ages=target_ages,
														   input_ages=input_ages,
														   no_aging=no_aging,
														   data_type="real")

				# perform cycle on generate images by setting the target ages to the original input ages
				y_hat_inverse = self.__set_target_to_source(x=y_hat, input_ages=input_ages)
				y_hat_inverse = torch.stack(y_hat_inverse)
				reverse_target_ages = y_hat_inverse[:, -1, 0, 0]
				y_recovered, latent_cycle = self.perform_forward_pass(y_hat_inverse)
				loss, cycle_loss_dict, cycle_id_logs = self.calc_loss(x, y, y_recovered, latent_cycle,
																	  target_ages=reverse_target_ages,
																	  input_ages=input_ages,
																	  no_aging=no_aging,
																	  data_type="cycle")

				# 鑑別器
				self.discriminatorloss = Conditionaldiscriminator_loss(self.discriminator, x, y_recovered)
				cur_loss_dict['discriminatorloss'] = self.discriminatorloss.item()
				loss += self.discriminatorloss * self.gan_loss_weight

				# combine the logs of both forwards
				for idx, cycle_log in enumerate(cycle_id_logs):
					id_logs[idx].update(cycle_log)
				cur_loss_dict.update(cycle_loss_dict)
				cur_loss_dict["loss"] = cur_loss_dict["loss_real"] + cur_loss_dict["loss_cycle"]

			agg_loss_dict.append(cur_loss_dict)

			# Logging related
			self.parse_and_log_images(id_logs, x, y, y_hat, y_recovered, title='images/test/faces',
									  subscript='{:04d}'.format(batch_idx))

			# For first step just do sanity test on small amount of data
			if self.global_step == 0 and batch_idx >= 4:
				self.net.train()
				return None  # Do not log, inaccurate in first batch

		loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
		self.log_metrics(loss_dict, prefix='test')
		self.print_metrics(loss_dict, prefix='test')

		self.net.train()
		return loss_dict

	def checkpoint_me(self, loss_dict, is_best):
		save_name = 'best_model.pt' if is_best else f'iteration_{self.global_step}.pt'
		save_dict = self.__get_save_dict()
		checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
		torch.save(save_dict, checkpoint_path)
		with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
			if is_best:
				f.write('**Best**: Step - {}, '
						'Loss - {:.3f} \n{}\n'.format(self.global_step, self.best_val_loss, loss_dict))
			else:
				f.write(f'Step - {self.global_step}, \n{loss_dict}\n')

	def configure_optimizers(self):
		params = list(self.net.encoder.parameters())
		if self.opts.train_decoder:
			params += list(self.net.decoder.parameters())
		if self.opts.optim_name == 'adam':
			optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
		else:
			optimizer = Ranger(params, lr=self.opts.learning_rate)
		return optimizer

	def configure_datasets(self):
		if self.opts.dataset_type not in data_configs.DATASETS.keys():
			Exception(f'{self.opts.dataset_type} is not a valid dataset_type')
		print(f'Loading dataset for {self.opts.dataset_type}')
		dataset_args = data_configs.DATASETS[self.opts.dataset_type]
		transforms_dict = dataset_args['transforms'](self.opts).get_transforms()
		train_dataset = ImagesDataset(source_root=dataset_args['train_source_root'],
									  target_root=dataset_args['train_target_root'],
									  source_transform=transforms_dict['transform_source'],
									  target_transform=transforms_dict['transform_gt_train'],
									  opts=self.opts)
		#train 只有人臉的照片


		test_dataset = ImagesDataset(source_root=dataset_args['test_source_root'],
									 target_root=dataset_args['test_target_root'],
									 source_transform=transforms_dict['transform_source'],
									 target_transform=transforms_dict['transform_test'],
									 opts=self.opts)
		#test 只有人臉的照片



		print(f"Number of training samples: {len(train_dataset)}")
		print(f"Number of test samples: {len(test_dataset)}")
		return train_dataset, test_dataset

	def calc_loss(self, x, y, y_hat, latent, target_ages, input_ages, no_aging, data_type="real"):
		loss_dict = {}
		id_logs = []
		loss = 0.0
		if self.opts.id_lambda > 0:
			weights = None
			if self.opts.use_weighted_id_loss:  # compute weighted id loss only on forward pass
				age_diffs = torch.abs(target_ages - input_ages)
				weights = train_utils.compute_cosine_weights(x=age_diffs)
			loss_id, sim_improvement, id_logs = self.id_loss(y_hat, y, x, label=data_type, weights=weights)
			loss_dict[f'loss_id_{data_type}'] = float(loss_id)
			loss_dict[f'id_improve_{data_type}'] = float(sim_improvement)
			loss = loss_id * self.opts.id_lambda
		if self.opts.l2_lambda > 0:
			loss_l2 = F.mse_loss(y_hat, y)
			loss_dict[f'loss_l2_{data_type}'] = float(loss_l2)
			if data_type == "real" and not no_aging:
				l2_lambda = self.opts.l2_lambda_aging
			else:
				l2_lambda = self.opts.l2_lambda
			loss += loss_l2 * l2_lambda
		if self.opts.lpips_lambda > 0:
			loss_lpips = self.lpips_loss(y_hat, y)
			loss_dict[f'loss_lpips_{data_type}'] = float(loss_lpips)
			if data_type == "real" and not no_aging:
				lpips_lambda = self.opts.lpips_lambda_aging
			else:
				lpips_lambda = self.opts.lpips_lambda
			loss += loss_lpips * lpips_lambda

		#shape loss
		if self.opts.shape_lambda > 0:
			#input_ages = self.aging_loss.extract_ages(y) / 100.
			#target_ages = self.aging_loss.extract_ages(y_hat) / 100.
			#print(target_ages.shape)
			#print(input_ages.shape)
			for i in range(target_ages.size(0)):
				#print(target_ages[i].item())
				points2_np = np.array(self.landmark_detector.detect_landmarks(common.tensor2im(y_hat[i])))

				if points2_np.shape == (9, 2):
					if target_ages[i].item() >= 0.06:
						points = [
							[71.68013468013469, 191.97643097643098],
							[82.41077441077441, 206.0841750841751],
							[96.77441077441077, 216.74410774410774],
							[113.73400673400674, 223.8114478114478],
							[133.2895622895623, 225.26599326599327],
							[152.36026936026937, 222.92592592592592],
							[168.32659932659934, 214.73063973063972],
							[181.76767676767676, 203.22222222222223],
							[191.33670033670035, 188.75420875420875]
						]
						total_distance = 0
						for j, point in enumerate(points):
							distance = np.linalg.norm(np.array(point) - points2_np[j])
							total_distance += distance

						loss_shape = total_distance / len(points)

						loss_dict[f'loss_shape_{data_type}'] = float(loss_shape)
						loss += loss_shape * self.opts.shape_lambda

					elif 0.05 <= target_ages[i].item() < 0.06:
						points = [
							[69.90132960111967, 190.93771868439467],
							[80.55843247025892, 204.6878936319104],
							[94.85199440167949, 215.1368089573128],
							[111.60566829951014, 222.1480055983205],
							[130.95066480055982, 223.80335899230232],
							[149.92162351294613, 221.52904128761372],
							[165.95976207137858, 213.61686494051784],
							[179.49825052484255, 202.51784464660602],
							[189.15850244926523, 188.44891532540237]
						]
						total_distance = 0
						for j, point in enumerate(points):
							distance = np.linalg.norm(np.array(point) - points2_np[j])
							total_distance += distance

						loss_shape = total_distance / len(points)

						loss_dict[f'loss_shape_{data_type}'] = float(loss_shape)
						loss += loss_shape * self.opts.shape_lambda

					elif 0.04 <= target_ages[i].item() < 0.05:
						points = [
							[69.28374400807333, 189.64157766377932],
							[80.30266588175931, 203.08771339668658],
							[94.90118577075098, 213.36826171053738],
							[111.70414599276764, 220.45681607938778],
							[130.9457572954335, 222.25918762088975],
							[149.80851063829786, 219.96106298881506],
							[165.86519216213944, 212.04389874695147],
							[179.57850475149272, 201.07375325876714],
							[189.4867546884198, 187.22950130350685]
						]
						total_distance = 0
						for j, point in enumerate(points):
							distance = np.linalg.norm(np.array(point) - points2_np[j])
							total_distance += distance

						loss_shape = total_distance / len(points)

						loss_dict[f'loss_shape_{data_type}'] = float(loss_shape)
						loss += loss_shape * self.opts.shape_lambda
					elif 0.03 <= target_ages[i].item() < 0.04:
						points = [
							[69.05979827089337, 188.0844249410532],
							[80.29329316216925, 201.26866649200943],
							[ 95.06850930049778, 211.36717317264868],
							[111.80586848310192, 218.4686926905947],
							[130.8068509300498, 220.37542572701074],
							[149.4405292114226, 218.15005239717055],
							[165.39409221902017, 210.39920094314908],
							[179.25340581608594, 199.6932800628766],
							[189.40011789363373, 186.1613177888394]
						]
						total_distance = 0
						for j, point in enumerate(points):
							distance = np.linalg.norm(np.array(point) - points2_np[j])
							total_distance += distance

						loss_shape = total_distance / len(points)

						loss_dict[f'loss_shape_{data_type}'] = float(loss_shape)
						loss += loss_shape * self.opts.shape_lambda

					elif 0.02 <= target_ages[i].item() < 0.03:
						points = [
							[70.16557841567081, 186.7749168411975],
							[81.30282123937415, 199.7205864235555],
							[95.85191573241346, 209.72767032154738],
							[112.29610693606013, 216.82450412714056],
							[131.0875323395343, 218.75920906738943],
							[149.54379696932364, 216.63373167426388],
							[165.22348158186523, 208.96864605149685],
							[178.8443390415178, 198.41610200813108],
							[188.9307625970186, 185.15060983121842]
						]
						total_distance = 0
						for j, point in enumerate(points):
							distance = np.linalg.norm(np.array(point) - points2_np[j])
							total_distance += distance

						loss_shape = total_distance / len(points)

						loss_dict[f'loss_shape_{data_type}'] = float(loss_shape)
						loss += loss_shape * self.opts.shape_lambda

					else:
						points = [
							[70.89358245329001, 188.16544814513946],
							[81.86731654481451, 201.18169509883563],
							[96.07229894394801, 211.37327376116977],
							[112.27267803953426, 218.64744110479285],
							[131.05970755483347, 220.60046033035474],
							[149.51854860546982, 218.4202545356079],
							[165.02355808285947, 210.57893311670728],
							[178.3734091524506, 199.8658272407257],
							[188.43203357703763, 186.57622529109125]
						]
						total_distance = 0
						for j, point in enumerate(points):
							distance = np.linalg.norm(np.array(point) - points2_np[j])
							total_distance += distance

						loss_shape = total_distance / len(points)

						loss_dict[f'loss_shape_{data_type}'] = float(loss_shape)
						loss += loss_shape * self.opts.shape_lambda

				else:
					print("Error: Unexpected shape for y_hat")




				#if target_ages[i].item() >= input_ages[i].item():  #(都是除100了)

				#	points1_np = np.array(self.landmark_detector.detect_landmarks(common.tensor2im(y[i])))
				#	points2_np = np.array(self.landmark_detector.detect_landmarks(common.tensor2im(y_hat[i])))

				#if target_ages[i].item() >= input_ages[i].item():  # (都是除100了)
					# target_ages - input_ages 可以分段
					# print('older')

					# points1_np = np.array(self.landmark_detector.detect_landmarks(common.tensor2im(y[i])))
				#	points2_np = np.array(self.landmark_detector.detect_landmarks(common.tensor2im(y_hat[i])))
					# print(points1_np.shape)  #(9, 2)
					# print(points2_np.shape)  #(9, 2)

				#	if points1_np.shape != (9, 2) or points2_np.shape != (9, 2):
				#		print("Error: Unexpected shape for points1_np or points2_np")

				#	else:
				#		d = np.linalg.norm(points1_np[1:] - points1_np[:-1], axis=1) - np.linalg.norm(
				#			points2_np[1:] - points2_np[:-1], axis=1)
				#		loss_shape = np.mean(np.abs(d))
				#		loss_dict[f'loss_shape_{data_type}'] = float(loss_shape)
				#		loss += loss_shape * self.opts.shape_lambda
				# else:
				# 	#如果年齡由大變小，增加額外的懲罰
				# 	#print('younger')
				# 	d5, d6 = self.landmark_detector.detect_landmarks(common.tensor2im(y[i]))  #common.tensor2im PIL image
				# 	d7, d8 = self.landmark_detector.detect_landmarks(common.tensor2im(y_hat[i]))
				# 	loss_shape = 1 / 2 * (abs(d5 - d7) + abs(d6 - d8))
				# 	penalty = self.opts.penalty_value * abs(input_ages[i].item() - target_ages[i].item())
				#
				# 	loss_dict[f'loss_shape_{data_type}'] = float(loss_shape)
				# 	loss += (loss_shape + penalty) * self.opts.shape_lambda

		if self.opts.color_lambda > 0:
			hair_weights = None
			#if target_ages[i].item() >= input_ages[i].item():
			for i in range(target_ages.size(0)):
				age_diffs = abs(target_ages[i] - input_ages[i])
				hair_weights = 0.5 * (np.sin(np.pi / 2 * age_diffs)) + 2
				#face_output_path = 'D:/hair/output.jpg'
				face_region = apply_segmentation_and_extract_hair(model, common.tensor2im(y_hat[i]), dataset.CLASSES)
				#Image.fromarray(face_region.astype(np.uint8)).save(face_output_path)

				#face_input_path = 'D:/hair/input.jpg'
				face_region2 = apply_segmentation_and_extract_hair(model, common.tensor2im(y[i]), dataset.CLASSES)
				#Image.fromarray(face_region2.astype(np.uint8)).save(face_input_path)

				loss_hist_sim = histogram_similarity_loss(face_region.astype(np.uint8),face_region2.astype(np.uint8))
				loss_dict[f'loss_hist_sim_{data_type}'] = float(loss_hist_sim)
				loss += hair_weights * loss_hist_sim * self.opts.color_lambda



		if self.opts.lpips_lambda_crop > 0:
			loss_lpips_crop = self.lpips_loss(y_hat[:, :, 35:223, 32:220], y[:, :, 35:223, 32:220])
			loss_dict['loss_lpips_crop'] = float(loss_lpips_crop)
			loss += loss_lpips_crop * self.opts.lpips_lambda_crop
		if self.opts.l2_lambda_crop > 0:
			loss_l2_crop = F.mse_loss(y_hat[:, :, 35:223, 32:220], y[:, :, 35:223, 32:220])
			loss_dict['loss_l2_crop'] = float(loss_l2_crop)
			loss += loss_l2_crop * self.opts.l2_lambda_crop


		if self.opts.w_norm_lambda > 0:
			loss_w_norm = self.w_norm_loss(latent, latent_avg=self.net.latent_avg)
			loss_dict[f'loss_w_norm_{data_type}'] = float(loss_w_norm)
			loss += loss_w_norm * self.opts.w_norm_lambda
		if self.opts.aging_lambda > 0:
			aging_loss, id_logs = self.aging_loss(y_hat, y, target_ages, id_logs, label=data_type)
			loss_dict[f'loss_aging_{data_type}'] = float(aging_loss)
			loss += aging_loss * self.opts.aging_lambda
		loss_dict[f'loss_{data_type}'] = float(loss)
		if data_type == "cycle":
			loss = loss * self.opts.cycle_lambda
		return loss, loss_dict, id_logs

	def log_metrics(self, metrics_dict, prefix):
		for key, value in metrics_dict.items():
			self.logger.add_scalar(f'{prefix}/{key}', value, self.global_step)

	def print_metrics(self, metrics_dict, prefix):
		print(f'Metrics for {prefix}, step {self.global_step}')
		for key, value in metrics_dict.items():
			print(f'\t{key} = ', value)

	def parse_and_log_images(self, id_logs, x, y, y_hat, y_recovered, title, subscript=None, display_count=2):
		im_data = []
		for i in range(display_count):  #min(display_count, len(x))原本是display_count
			cur_im_data = {
				'input_face': common.tensor2im(x[i]),   #輸出圖片
				'target_face': common.tensor2im(y[i]),
				'output_face': common.tensor2im(y_hat[i]),
				'recovered_face': common.tensor2im(y_recovered[i])
			}
			if id_logs is not None:
				for key in id_logs[i]:
					cur_im_data[key] = id_logs[i][key]
			im_data.append(cur_im_data)
		self.log_images(title, im_data=im_data, subscript=subscript)

	def log_images(self, name, im_data, subscript=None, log_latest=False):
		fig = common.vis_faces(im_data)
		step = self.global_step
		if log_latest:
			step = 0
		if subscript:
			path = os.path.join(self.logger.log_dir, name, '{}_{:04d}.jpg'.format(subscript, step))
		else:
			path = os.path.join(self.logger.log_dir, name, '{:04d}.jpg'.format(step))
		os.makedirs(os.path.dirname(path), exist_ok=True)
		fig.savefig(path)
		plt.close(fig)

	def __get_save_dict(self):
		save_dict = {
			'state_dict': self.net.state_dict(),
			'opts': vars(self.opts)
		}
		# save the latent avg in state_dict for inference if truncation of w was used during training
		if self.net.latent_avg is not None:
			save_dict['latent_avg'] = self.net.latent_avg
		return save_dict

	def save_model(self, dir, filename):     #加給鑑別器用
		torch.save(self.discriminator.state_dict(), os.path.join(dir, "d" + filename))



