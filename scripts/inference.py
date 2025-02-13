from argparse import Namespace
import os
import time
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from datasets.augmentations import AgeTransformer
from utils.common import tensor2im, log_image
from options.test_options import TestOptions
from models.psp import pSp

#在這邊做語義分割
from Extract import *
import os

def run():
	test_opts = TestOptions().parse()

	out_path_results = os.path.join(test_opts.exp_dir, 'inference_results')
	out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled')
	os.makedirs(out_path_results, exist_ok=True)
	os.makedirs(out_path_coupled, exist_ok=True)

	# update test options with options used during training
	ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
	opts = ckpt['opts']
	opts.update(vars(test_opts))
	opts = Namespace(**opts)

	net = pSp(opts)
	net.eval()
	net.cuda()

	age_transformers = [AgeTransformer(target_age=age) for age in opts.target_age.split(',')]

	print(f'Loading dataset for {opts.dataset_type}')
	dataset_args = data_configs.DATASETS[opts.dataset_type]
	transforms_dict = dataset_args['transforms'](opts).get_transforms()

	#在這邊做人臉語義分割
	# 下載模型
	download_models()

	# 載入模型
	resolution = 256
	assert torch.cuda.is_available()
	torch.backends.cudnn.benchmark = True
	model_fname = 'deeplab_model/deeplab_model.pth'
	# inference
	dataset_root = opts.data_path  # 裡面再一個資料夾 再放圖片
	assert os.path.isdir(dataset_root)
	dataset = CelebASegmentation(dataset_root, crop_size=256)

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

	os.makedirs(opts.mask_path, exist_ok=True)
	# 應用語義分割
	apply_semantic_segmentation(model, dataset, save_folder=opts.mask_path)

	os.makedirs(opts.background_path, exist_ok=True)
	os.makedirs(opts.face_path, exist_ok=True)


	segmentation_folder = opts.mask_path
	original_images_folder = opts.orig_img  #可能有錯,因裡面還有個資料夾 可以新弄個 opts.orig_img
	face_folder = opts.face_path
	background_output_folder = opts.background_path



	if not os.path.exists(face_folder):
		os.makedirs(face_folder)
	if not os.path.exists(background_output_folder):
		os.makedirs(background_output_folder)

	# 提取人臉和背景
	extract_faceandbackground(segmentation_folder, original_images_folder, face_folder,background_output_folder) #結合後面做


	dataset = InferenceDataset(root=face_folder,
							   transform=transforms_dict['transform_inference'],
							   opts=opts)
	dataloader = DataLoader(dataset,
							batch_size=opts.test_batch_size,
							shuffle=False,
							num_workers=int(opts.test_workers),
							drop_last=False)

	# background_output_folder弄成1024*1024  segmentation_folder也要弄成
	resize_amount = (256, 256) if opts.resize_outputs else (1024, 1024)
	#背景
	for filename in os.listdir(background_output_folder):
		if filename.endswith(".jpg") or filename.endswith(".png"):
			image_path = os.path.join(background_output_folder, filename)
			image = Image.open(image_path)
			# 調整大小BICUBIC插值
			resized_image = image.resize(resize_amount, Image.BICUBIC)
			resized_image.save(image_path.replace(".jpg", ".png"))
	#mask
	for filename in os.listdir(segmentation_folder):
		if filename.endswith(".jpg") or filename.endswith(".png"):
			image_path = os.path.join(segmentation_folder, filename)
			image = Image.open(image_path)
			resized_image = image.resize(resize_amount, Image.BICUBIC)
			resized_image.save(image_path.replace(".jpg", ".png"))


	if opts.n_images is None:
		opts.n_images = len(dataset)

	global_time = []
	for age_transformer in age_transformers:
		print(f"Running on target age: {age_transformer.target_age}")
		global_i = 0
		for input_batch in tqdm(dataloader):
			if global_i >= opts.n_images:
				break
			with torch.no_grad():
				input_age_batch = [age_transformer(img.cpu()).to('cuda') for img in input_batch]
				input_age_batch = torch.stack(input_age_batch)
				input_cuda = input_age_batch.cuda().float()
				tic = time.time()
				result_batch = run_on_batch(input_cuda, net, opts)
				toc = time.time()
				global_time.append(toc - tic)

				for i in range(len(input_batch)):
					result = tensor2im(result_batch[i])
					im_path = dataset.paths[global_i]

					if opts.couple_outputs or global_i % 100 == 0:
						input_im = log_image(input_batch[i], opts)
						resize_amount = (256, 256) if opts.resize_outputs else (1024, 1024)
						res = np.concatenate([np.array(input_im.resize(resize_amount)),
											  np.array(result.resize(resize_amount))], axis=1)
						age_out_path_coupled = os.path.join(out_path_coupled, age_transformer.target_age)
						os.makedirs(age_out_path_coupled, exist_ok=True)
						Image.fromarray(res).save(os.path.join(age_out_path_coupled, os.path.basename(im_path)))

					age_out_path_results = os.path.join(out_path_results, age_transformer.target_age)
					os.makedirs(age_out_path_results, exist_ok=True)

					#age_out_path_results 輸出的是  生成人臉圖片位址

					image_name = os.path.basename(im_path)
					im_save_path = os.path.join(age_out_path_results, image_name.replace(".jpg", ".png"))  #人臉
					Image.fromarray(np.array(result.resize(resize_amount))).save(im_save_path.replace(".jpg", ".png"))

					combine_face_and_background(im_save_path, background_output_folder, segmentation_folder)
					global_i += 1


	stats_path = os.path.join(opts.exp_dir, 'stats.txt')
	result_str = 'Runtime {:.4f}+-{:.4f}'.format(np.mean(global_time), np.std(global_time))
	print(result_str)

	with open(stats_path, 'w') as f:
		f.write(result_str)


def run_on_batch(inputs, net, opts):
	result_batch = net(inputs, randomize_noise=False, resize=opts.resize_outputs)
	return result_batch


if __name__ == '__main__':
	run()
