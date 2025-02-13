import argparse
import os
import requests
import numpy as np
import torch
import torch.nn as nn
from pdb import set_trace as st
from PIL import Image
from torchvision import transforms
from PIL import ImageFilter
import cv2


import deeplab
from data_loader import CelebASegmentation
from utils2 import download_file


class_color_map = {
    'background': (0, 0, 0),
    'skin': (0, 0, 0),
    'nose': (0, 0, 0),
    'eye_g': (0, 0, 0),
    'l_eye': (0, 0, 0),
    'r_eye': (0, 0, 0),
    'l_brow': (255, 255, 255),
    'r_brow': (255, 255, 255),
    'l_ear': (0, 0, 0),
    'r_ear': (0, 0, 0),
    'mouth': (0, 0, 0),
    'u_lip': (0, 0, 0),
    'l_lip': (0, 0, 0),
    'hair': (255, 255, 255),
    'hat': (0, 0, 0),
    'ear_r': (0, 0, 0),
    'neck_l': (0, 0, 0),
    'neck': (0, 0, 0),
    'cloth': (0, 0, 0)
}

#parser = argparse.ArgumentParser()
#parser.add_argument('--resolution', type=int, default=256,help='segmentation output size')
#parser.add_argument('--workers', type=int, default=4,help='number of data loading workers')
#args = parser.parse_args()

resnet_file_spec = dict(file_url='https://drive.google.com/uc?id=1oRGgrI4KNdefbWVpw0rRkEP1gbJIRokM', file_path='deeplab_model/R-101-GN-WS.pth.tar', file_size=178260167, file_md5='aa48cc3d3ba3b7ac357c1489b169eb32')
deeplab_file_spec = dict(file_url='https://drive.google.com/uc?id=1w2XjDywFr2NjuUWaLQDRktH7VwIfuNlY', file_path='deeplab_model/deeplab_model.pth', file_size=464446305, file_md5='8e8345b1b9d95e02780f9bed76cc0293')



def download_models():
    # 下載 ResNet 模型參數
    if not os.path.isfile(resnet_file_spec['file_path']):
        print('Downloading backbone Resnet Model parameters')
        with requests.Session() as session:
            download_file(session, resnet_file_spec)
        print('Done!')

    # 下載 DeeplabV3 模型參數
    if not os.path.isfile(deeplab_file_spec['file_path']):
        print('Downloading DeeplabV3 Model parameters')
        with requests.Session() as session:
            download_file(session, deeplab_file_spec)
        print('Done!')



def apply_segmentation_and_extract_hair(model, pil_img, classes):
    #if not os.path.exists(os.path.dirname(face_output_path)):
    #    os.makedirs(os.path.dirname(face_output_path))
    img = np.array(pil_img)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    inputs = transform(img).unsqueeze(0).cuda()
    outputs = model(inputs)
    _, pred = torch.max(outputs, 1)
    pred = pred.data.cpu().numpy().squeeze().astype(np.uint8)

    # 将灰度图转换为彩色图像
    mask_color = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    for j, class_name in enumerate(classes):
        mask_color[pred == j] = class_color_map[class_name]

    # 将分割结果转换为NumPy数组
    segmentation_array = mask_color

    # 頭髮（255, 255, 255），背景为黑色（0, 0, 0）
    hair_mask = (segmentation_array >= [128, 128, 128]).all(axis=-1).astype(np.uint8)

    # 提取人脸区域
    original_array = np.array(img)
    hair_region = original_array * np.expand_dims(hair_mask, axis=2)
    non_hair_color = [0, 0, 0]  # 黑色
    hair_region = hair_region + np.expand_dims(1 - hair_mask, axis=2) * np.array(non_hair_color)


    return hair_region

    # 保存提取的人脸区域
    #Image.fromarray(face_region.astype(np.uint8)).save(face_output_path)
    #print(f'Processed and saved to {face_output_path}')


def histogram_similarity_loss(image1, image2):
    #  HSV
    hsv_image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2HSV)
    hsv_image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2HSV)

    # 顏色直方圖
    hist1 = cv2.calcHist([hsv_image1], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist2 = cv2.calcHist([hsv_image2], [0, 1], None, [50, 60], [0, 180, 0, 256])

    # normalize
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()

    #  Bhattacharyya 距離
    loss = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    return -loss





# def main():
#     # 下载模型
#     download_models()
#
#     # 加载模型
#     resolution = 256
#     assert torch.cuda.is_available()
#     torch.backends.cudnn.benchmark = True
#     model_fname = 'deeplab_model/deeplab_model.pth'
#     dataset_root = '/home/tony/img'
#     assert os.path.isdir(dataset_root)
#     dataset = CelebASegmentation(dataset_root, crop_size=256)
#
#     model = getattr(deeplab, 'resnet101')(
#         pretrained=True,
#         num_classes=len(dataset.CLASSES),
#         num_groups=32,
#         weight_std=True,
#         beta=False)
#
#     model = model.cuda()
#     model.eval()
#     checkpoint = torch.load(model_fname)
#     state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k}
#     model.load_state_dict(state_dict)
#
#     # 读取输入图像
#     pil_img = Image.open('/home/tony/img/image/00059.jpg')
#     #img = np.array(pil_img)
#
#
#
#     # 应用语义分割并提取人脸区域
#     face_output_path = '/home/tony/face/00059.jpg'
#     face_region = apply_segmentation_and_extract_hair(model, pil_img, dataset.CLASSES)
#     Image.fromarray(face_region.astype(np.uint8)).save(face_output_path)