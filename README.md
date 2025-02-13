# AGE

# Environment
- Python 3.8.19
- pytorch 1.8.1 + cu111


## Table of Contents
  * [Getting Started](#getting-started)
    + [Prerequisites](#prerequisites)
    + [Installation](#installation)
  * [Pretrained Models](#pretrained-models)
  * [Training](#training)
    + [Preparing your Data](#preparing-your-data)
    + [Training model](#training-sam)
    + [Additional Notes](#additional-notes)
  * [Notebooks](#notebooks)
    + [Inference Notebook](#inference-notebook)
  * [Testing](#testing)
    + [Inference](#inference)
  * [Repository structure](#repository-structure)
  * [Credits](#credits)
  * [Acknowledgments](#acknowledgments)
  * [Citation](#citation)


## Getting Started
### Prerequisites
- Linux or macOS
- NVIDIA GPU + CUDA CuDNN (CPU may be possible with some modifications, but is not inherently supported)
- Python 3

### Installation
- Dependencies:  
We recommend running this repository using [Anaconda](https://docs.anaconda.com/anaconda/install/). 
All dependencies for defining the environment are provided in `environment/sam_env.yaml`.

## Pretrained Models
Please download the pretrained aging model from the following links.

| Path | Description
| :--- | :----------
|[AGE]|[(https://drive.google.com/file/d/1XpjQHfivo2eQg2Cj5ZjmIS6v2rBpXD0G/view?usp=drive_link) ] | .

download:
```
mkdir pretrained_models
pip install gdown
gdown "https://drive.google.com/u/0/uc?id=1XyumF6_fdAxFmxpFcmPf-q84LU_22EMC&export=download" -O pretrained_models/sam_ffhq_aging.pt
wget "https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat"
```

Pretrained_models

| Path | Description
| :--- | :----------
|[pSp Encoder](https://drive.google.com/file/d/1bMTNWkh5LArlaWSc_wa8VKyq2V42T2z0/view?usp=sharing) | pSp taken from [pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel) trained on the FFHQ dataset for StyleGAN inversion.
|[FFHQ StyleGAN](https://drive.google.com/file/d/1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT/view?usp=sharing) | StyleGAN model pretrained on FFHQ taken from [rosinality](https://github.com/rosinality/stylegan2-pytorch) with 1024x1024 output resolution.
|[IR-SE50 Model](https://drive.google.com/file/d/1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn/view?usp=sharing) | Pretrained IR-SE50 model taken from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch) for use in our ID loss during training.
|[VGG Age Classifier](https://drive.google.com/file/d/1atzjZm_dJrCmFWCqWlyspSpr3nI6Evsh/view?usp=sharing) | VGG age classifier from DEX and fine-tuned on the FFHQ-Aging dataset for use in our aging loss

By default, we assume that all auxiliary models are downloaded and saved to the directory `pretrained_models` and change `configs/path_configs.py`. 

## Training
### Preparing your Data
Please refer to `configs/paths_config.py` to define the necessary data paths and model paths for training and inference.   
Then, refer to `configs/data_configs.py` to define the source/target data paths for the train and test sets as well as the 
transforms to be used for training and inference.
    
As an example, we can first go to `configs/paths_config.py` and define:
``` 
dataset_paths = {
    'ffhq': '/path/to/ffhq/images256x256'
    'celeba_test': '/path/to/CelebAMask-HQ/test_img',
}
```
Then, in `configs/data_configs.py`, we define:
``` 
DATASETS = {
	'ffhq_aging': {
		'transforms': transforms_config.AgingTransforms,
		'train_source_root': dataset_paths['ffhq'],
		'train_target_root': dataset_paths['ffhq'],
		'test_source_root': dataset_paths['test'],
		'test_target_root': dataset_paths['test'],
	}
}
``` 
When defining the datasets for training and inference, we will use the values defined in the above dictionary.


### Training
The main training script can be found in `scripts/train.py`.   
Intermediate training results are saved to `opts.exp_dir`. This includes checkpoints, train outputs, and test outputs.  
Additionally, if you have tensorboard installed, you can visualize tensorboard logs in `opts.exp_dir/logs`.

Training SAM with the settings used in the paper can be done by running the following command:
```
python scripts/train.py \
--dataset_type=ffhq_aging \
--exp_dir=/path/to/experiment \
--workers=6 \
--batch_size=6 \
--test_batch_size=6 \
--test_workers=6 \
--val_interval=2500 \
--save_interval=10000 \
--start_from_encoded_w_plus \
--id_lambda=0.1 \
--lpips_lambda=0.1 \
--lpips_lambda_aging=0.1 \
--lpips_lambda_crop=0.6 \
--l2_lambda=0.25 \
--l2_lambda_aging=0.25 \
--l2_lambda_crop=1 \
--w_norm_lambda=0.005 \
--aging_lambda=5 \
--cycle_lambda=1 \
--input_nc=4 \
--target_age=uniform_random \
--use_weighted_id_loss
```

### Additional Notes
- See `options/train_options.py` for all training-specific flags. 
- Note that using the flag `--start_from_encoded_w_plus` requires you to specify the path to the pretrained pSp encoder.  
    By default, this path is taken from `configs.paths_config.model_paths['pretrained_psp']`.
- If you wish to resume from a specific checkpoint (e.g. a pretrained SAM model), you may do so using `--checkpoint_path`.



## Testing
### Inference
Having trained your model or if you're using a pretrained SAM model, you can use `scripts/inference.py` to run inference
on a set of images.   
For example, 
```
python scripts/inference.py \
--exp_dir=/path/to/experiment \
--checkpoint_path=experiment/checkpoints/best_model.pt \
--data_path=/path/to/test_data \
--test_batch_size=4 \
--test_workers=4 \
--couple_outputs
--target_age=0,10,20,30,40,50,60,70,80
```
Additional notes to consider: 
- During inference, the options used during training are loaded from the saved checkpoint and are then updated using the 
test options passed to the inference script.
- Adding the flag `--couple_outputs` will save an additional image containing the input and output images side-by-side in the sub-directory
`inference_coupled`. Otherwise, only the output image is saved to the sub-directory `inference_results`.
- In the above example, we will run age transformation with target ages 0,10,...,80.
    - The results of each target age are saved to the sub-directories `inference_results/TARGET_AGE` and `inference_coupled/TARGET_AGE`.
- By default, the images will be saved at resolution of 1024x1024, the original output size of StyleGAN. 
    - If you wish to save outputs resized to resolutions of 256x256, you can do so by adding the flag `--resize_outputs`.

### Side-by-Side Inference 
The above inference script will save each aging result in a different sub-directory for each target age. Sometimes, 
however, it is more convenient to save all aging results of a given input side-by-side like the following: 

<p align="center">
<img src="docs/866.jpg" width="800px"/>
</p>

To do so, we provide a script `inference_side_by_side.py` that works in a similar manner as the regular inference script: 
```
python scripts/inference_side_by_side.py \
--exp_dir=/path/to/experiment \
--checkpoint_path=experiment/checkpoints/best_model.pt \
--data_path=/path/to/test_data \
--test_batch_size=4 \
--test_workers=4 \
--target_age=0,10,20,30,40,50,60,70,80
```
Here, all aging results 0,10,...,80 will be save side-by-side with the original input image.

## Repository structure
| Path | Description <img width=200>
| :--- | :---
| SAM | Repository root folder
| &boxvr;&nbsp; configs | Folder containing configs defining model/data paths and data transforms
| &boxvr;&nbsp; criteria | Folder containing various loss criterias for training
| &boxvr;&nbsp; datasets | Folder with various dataset objects and augmentations
| &boxvr;&nbsp; docs | Folder containing images displayed in the README
| &boxvr;&nbsp; environment | Folder containing Anaconda environment used in our experiments
| &boxvr; models | Folder containing all the models and training objects
| &boxv;&nbsp; &boxvr;&nbsp; encoders | Folder containing various architecture implementations
| &boxv;&nbsp; &boxvr;&nbsp; stylegan2 | StyleGAN2 model from [rosinality](https://github.com/rosinality/stylegan2-pytorch)
| &boxv;&nbsp; &boxvr;&nbsp; psp.py | Implementation of pSp encoder
| &boxv;&nbsp; &boxur;&nbsp; dex_vgg.py | Implementation of DEX VGG classifier used in computation of aging loss
| &boxvr;&nbsp; notebook | Folder with jupyter notebook containing SAM inference playground
| &boxvr;&nbsp; options | Folder with training and test command-line options
| &boxvr;&nbsp; scripts | Folder with running scripts for training and inference
| &boxvr;&nbsp; training | Folder with main training logic and Ranger implementation from [lessw2020](https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer)
| &boxvr;&nbsp; utils | Folder with various utility functions
| <img width=300> | <img>


## Credits
**StyleGAN2 model and implementation:**  
https://github.com/rosinality/stylegan2-pytorch  
Copyright (c) 2019 Kim Seonghyeon  
License (MIT) https://github.com/rosinality/stylegan2-pytorch/blob/master/LICENSE  

**IR-SE50 model and implementations:**  
https://github.com/TreB1eN/InsightFace_Pytorch  
Copyright (c) 2018 TreB1eN  
License (MIT) https://github.com/TreB1eN/InsightFace_Pytorch/blob/master/LICENSE  

**Ranger optimizer implementation:**  
https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer   
License (Apache License 2.0) https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer/blob/master/LICENSE  

**LPIPS model and implementation:**  
https://github.com/S-aiueo32/lpips-pytorch  
Copyright (c) 2020, Sou Uchida  
License (BSD 2-Clause) https://github.com/S-aiueo32/lpips-pytorch/blob/master/LICENSE  

**DEX VGG model and implementation:**  
https://github.com/InterDigitalInc/HRFAE  
Copyright (c) 2020, InterDigital R&D France  
https://github.com/InterDigitalInc/HRFAE/blob/master/LICENSE.txt

**pSp model and implementation:**   
https://github.com/eladrich/pixel2style2pixel  
Copyright (c) 2020 Elad Richardson, Yuval Alaluf  
https://github.com/eladrich/pixel2style2pixel/blob/master/LICENSE

## Acknowledgments
This code borrows heavily from [pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel)

## Citation
If you use this code for your research, please cite our paper :

```
@inproceedings{hsieh2024agesynthgan,
  title={AgeSynthGAN: Advanced Facial Age Synthesis with StyleGAN2},
  author={Hsieh, Tung-Ke and Liu, Tsung-Jung and Liu, Kuan-Hsien},
  booktitle={2024 IEEE International Conference on Visual Communications and Image Processing (VCIP)},
  pages={1--5},
  year={2024},
  organization={IEEE}
}
```



