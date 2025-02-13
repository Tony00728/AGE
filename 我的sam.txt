# CUDA 11.0
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch

dataset : CelebAMask-HQ 在F:
	  FFHQ 在F:
-----------

dataset_paths = {
    'celeba_test': '/test_img',
    'ffhq': '/train/009055_0M54.jpg',
}
---------
dataset_paths = {
    'celeba_test': '/path/to/CelebAMask-HQ/test_img',
    'ffhq': '/path/to/ffhq/images256x256',
}


----------------------------------

python scripts/inference.py \
--exp_dir(保存輸出的位置)=/path/to/experiment \
--checkpoint_path(模型位置)=experiment/checkpoints/best_model.pt \
--data_path(圖片)=/path/to/test_data \
--test_batch_size=4 \
--test_workers=4 \
--couple_outputs
--target_age=0,10,20,30,40,50,60,70,80

-------------------------------------------------------------------------------
https://blog.csdn.net/jorg_zhao/article/details/106883420 環境

--------------------test

linux的test
python /home/tony/SAM-master/scripts/inference.py --exp_dir=/home/tony/SAM-master/experiment/ --checkpoint_path=/home/tony/SAM-master/pretrained_models/sam_ffhq_aging.pt --data_path= /home/tony/SAM-master/CelebAMask-HQ/test_img/ --test_batch_size=2 --test_workers=1 --couple_outputs --target_age=0,10,20,30,40,50,60,70,80

---------
跑學長論文的圖 ，這可以
python /home/tony/SAM-master/scripts/inference.py --exp_dir=/home/tony/SAM-master/experiment/ --checkpoint_path=/home/tony/SAM-master/pretrained_models/sam_ffhq_aging.pt --data_path=/home/tony/SAM-master/paper_fig/ --test_batch_size=2 --test_workers=1 --couple_outputs --target_age=0,10,20,30,40,50,60,70,80



Training--------    dataset_type=ffhq_aging(這在configs/data_configs.py)
python scripts/train.py --dataset_type=ffhq_aging  --exp_dir=experiment --workers=4 --batch_size=4 --test_batch_size=4 --test_workers=4 --val_interval=200 --save_interval=1000 --start_from_encoded_w_plus --id_lambda=0.1 --lpips_lambda=0.1 --lpips_lambda_aging=0.1 --lpips_lambda_crop=0.6 --l2_lambda=0.25 --l2_lambda_aging=0.25 --l2_lambda_crop=1 --w_norm_lambda=0.005 --aging_lambda=5 --cycle_lambda=1 --input_nc=4 --target_age=uniform_random --use_weighted_id_loss

----這可以
python scripts/train.py --dataset_type=ffhq_aging  --exp_dir=experiment --workers=4 --batch_size=4 --test_batch_size=4 --test_workers=4 --val_interval=200 --save_interval=1000 --start_from_encoded_w_plus --id_lambda=0.1 --lpips_lambda=0.1 --lpips_lambda_aging=0.1 --lpips_lambda_crop=0.6 --l2_lambda=0.25 --l2_lambda_aging=0.25 --l2_lambda_crop=1 --w_norm_lambda=0.005 --aging_lambda=5 --cycle_lambda=1 --input_nc=4 --target_age=uniform_random --use_weighted_id_loss

----------------------
(sam_env2) tony@tony-System-Product-Name:~/SAM-master$ python scripts/train.py --dataset_type=ffhq_aging  --exp_dir=experiment --workers=2 --batch_size=2 --test_batch_size=2 --test_workers=2 --val_interval=200 --save_interval=1000 --start_from_encoded_w_plus --id_lambda=0.1 --lpips_lambda=0.1 --lpips_lambda_aging=0.1 --lpips_lambda_crop=0.6 --l2_lambda=0.25 --l2_lambda_aging=0.25 --l2_lambda_crop=1 --w_norm_lambda=0.005 --aging_lambda=5 --cycle_lambda=1 --input_nc=4 --target_age=uniform_random --use_weighted_id_loss

--------------------
增加workers的數量 讓GPU使用率變高

Dataset 要256x256  .jpg

--------
tensorboard --logdir=logs

python scripts/train.py --dataset_type=ffhq_aging  --exp_dir=experiment_face3 --workers=4 --batch_size=4 --test_batch_size=4 --test_workers=4 --val_interval=2500 --save_interval=10000 --start_from_encoded_w_plus --id_lambda=0.2 --lpips_lambda=0.7 --lpips_lambda_aging=0.2 --l2_lambda=1 --l2_lambda_aging=0.25 --w_norm_lambda=0.005 --aging_lambda=6 --cycle_lambda=1.2 --input_nc=4 --target_age=uniform_random --use_weighted_id_loss


morph train
python scripts/train.py --dataset_type=morph_dataset  --exp_dir=MORPH_exp --workers=4 --batch_size=4 --test_batch_size=4 --test_workers=4 --val_interval=2500 --save_interval=10000 --start_from_encoded_w_plus --id_lambda=0.2 --lpips_lambda=0.7 --lpips_lambda_aging=0.2 --l2_lambda=1 --l2_lambda_aging=0.25 --w_norm_lambda=0.005 --aging_lambda=6 --cycle_lambda=1.2 --input_nc=4 --target_age=uniform_random --use_weighted_id_loss

FFHQ
python scripts/train.py --dataset_type=ffhq_aging  --exp_dir=exp_plusimg --workers=4 --batch_size=4 --test_batch_size=4 --test_workers=4 --val_interval=2500 --save_interval=10000 --start_from_encoded_w_plus --id_lambda=0.2 --lpips_lambda=0.7 --lpips_lambda_aging=0.2 --l2_lambda=1 --l2_lambda_aging=0.25 --w_norm_lambda=0.005 --aging_lambda=6 --cycle_lambda=1.2 --input_nc=4 --target_age=uniform_random --use_weighted_id_loss


最新train用這個
python scripts/train.py --dataset_type=ffhq_aging  --exp_dir=exp_sam_fading --workers=4 --batch_size=4 --test_batch_size=4 --test_workers=4 --val_interval=2500 --save_interval=10000 --start_from_encoded_w_plus --id_lambda=0.2 --lpips_lambda=0.7 --lpips_lambda_aging=0.2 --l2_lambda=1 --l2_lambda_aging=0.25 --w_norm_lambda=0.005 --aging_lambda=6 --cycle_lambda=1.2 --input_nc=4 --target_age=uniform_random --use_weighted_id_loss



-----分割後的test
python /home/tony/SAM-master/scripts/inference.py --exp_dir=/home/tony/exp/ --checkpoint_path=/home/tony/SAM-master/experiment_face/best_model.pt --data_path=/home/tony/30-39/ --orig_img=/home/tony/30-39/image --mask_path=/home/tony/mask --background_path=/home/tony/background --face_path=/home/tony/face --test_batch_size=4 --test_workers=2 --target_age=0,20
python /home/tony/SAM-master/scripts/inference.py --exp_dir=/home/tony/test/ --checkpoint_path=/home/tony/SAM-master/exp_ffhq_shapeloss1/best_model.pt --data_path=/home/tony/CelebAMask-HQ/ --orig_img=/home/tony/CelebAMask-HQ/test_img_resize --mask_path=/home/tony/mask --background_path=/home/tony/background --face_path=/home/tony/face --test_batch_size=4 --test_workers=2 --target_age=0,20,35,48,58,68,80 --resize_outputs

python /home/tony/SAM-master/scripts/inference.py --exp_dir=/home/tony/exp/shapeloss/40-49/ --checkpoint_path=/home/tony/SAM-master/experiment_face5/best_model.pt --data_path=/home/tony/FFHQ-Aging_testing/40-49/ --orig_img=/home/tony/FFHQ-Aging_testing/40-49/image --mask_path=/home/tony/mask --background_path=/home/tony/background --face_path=/home/tony/face --test_batch_size=4 --test_workers=2 --target_age=0,25,35,45,55,65,80 --resize_outputs
id_loss有crop弄掉
加 --resize_outputs  輸出為256*256
python scripts/inference.py --exp_dir=C:/exp_no_arcface/20-/ --checkpoint_path=D:\AGE-master/exp_no_arcface/checkpoints/best_model.pt --data_path=C:/FFHQ-Aging_testing/10-14/ --orig_img=C:/FFHQ-Aging_testing/10-14/image --mask_path=C:/mask --background_path=C:/background --face_path=C:/face --test_batch_size=4 --test_workers=2 --target_age=0,20,35,45,55,65,75 --resize_outputs

------------------------
語義分割
Train: 改了/home/tony/SAM-master/configs/paths_config.py   Extract.py  語義分割的utils.py 變成utils2.py

/home/tony/SAM-master/semantic_segmentation_done.txt 這個換資料集要刪掉 有關/home/tony/SAM-master/configs/paths_config.py

Testing: 改了 inference.py跟test_options.py ,  paths_config.py的dataset_paths要改


---------------


coach_aging.py裡面這可以不用

		if self.opts.lpips_lambda_crop > 0:
			loss_lpips_crop = self.lpips_loss(y_hat[:, :, 35:223, 32:220], y[:, :, 35:223, 32:220])
			loss_dict['loss_lpips_crop'] = float(loss_lpips_crop)
			loss += loss_lpips_crop * self.opts.lpips_lambda_crop
		if self.opts.l2_lambda_crop > 0:
			loss_l2_crop = F.mse_loss(y_hat[:, :, 35:223, 32:220], y[:, :, 35:223, 32:220])
			loss_dict['loss_l2_crop'] = float(loss_l2_crop)
			loss += loss_l2_crop * self.opts.l2_lambda_crop



-----------Morph
資料集圖片要是 .jpg
python /home/tony/SAM-master/scripts/inference.py --exp_dir=/home/tony/exp/morph/shapeloss/20-29 --checkpoint_path=/home/tony/SAM-master/morph_exp_shapeloss/best_model.pt --data_path=/home/tony/morph_testing/20-29 --orig_img=/home/tony/morph_testing/20-29/image --mask_path=/home/tony/mask --background_path=/home/tony/background --face_path=/home/tony/face --test_batch_size=4 --test_workers=2 --target_age=0,20,35,45,55,65,80 --resize_outputs



----------FADING
conda install -c conda-forge diffusers
pip install --upgrade huggingface_hub

先生成Fading image
dataset的augmentations 改目標年齡範圍
gender male 或是 females
第一步 :python fading_age_editing.py --image_folder D:/AGE-master/ffhq_asian_man/image --age_init None --gender male --save_aged_dir D:/FADING-master/asian_man --specialized_path D:/FADING-master/finetune_double_prompt_150_random --target_ages 10 80
第二步 :D:\SAM-master\Extract_face.py

改training/coach_aging.py 的FADING路徑 跟configs/paths_config.py路徑

----
改了D:\SAM-master\.\training\ranger.py:123: UserWarning: This overload of addcmul_ is deprecated:addcmul_(Number value, Tensor tensor1, Tensor tensor2)
Consider using one of the following signatures instead:addcmul_(Tensor tensor1, Tensor tensor2, *, Number value) (Triggered internally at  ..\torch\csrc\utils\python_arg_parser.cpp:1005.)exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

改動UserWarning: Default upsampling behavior when mode=bilinear is changed to align_corners=False since 0.4.0. Please specify align_corners=True if the old behavior is desired. See the documentation of nn.Upsample for details.
\criteria\aging_loss.py


train
python scripts/train.py --dataset_type=ffhq_aging  --exp_dir=exp_fading_shapeloss_black_females --workers=4 --batch_size=4 --test_batch_size=4 --test_workers=4 --val_interval=2500 --save_interval=10000 --start_from_encoded_w_plus --id_lambda=0.2 --lpips_lambda=0.7 --lpips_lambda_aging=0.2 --l2_lambda=1 --l2_lambda_aging=0.25 --w_norm_lambda=0.005 --aging_lambda=6 --fading_lambda=3 --cycle_lambda=1.2 --input_nc=4 --target_age=uniform_random --use_weighted_id_loss
