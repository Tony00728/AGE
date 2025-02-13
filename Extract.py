import argparse
import os
import requests
import numpy as np
import torch
import torch.nn as nn
from pdb import set_trace as st
from PIL import Image
from torchvision import transforms

import deeplab
from data_loader import CelebASegmentation
from utils2 import download_file


# 自定義顏色映射
class_color_map = {
    'background': (0, 0, 0),
    'skin': (255, 255, 255),
    'nose': (255, 255, 255),
    'eye_g': (255, 255, 255),
    'l_eye': (255, 255, 255),
    'r_eye': (255, 255, 255),
    'l_brow': (255, 255, 255),
    'r_brow': (255, 255, 255),
    'l_ear': (255, 255, 255),
    'r_ear': (255, 255, 255),
    'mouth': (255, 255, 255),
    'u_lip': (255, 255, 255),
    'l_lip': (255, 255, 255),
    'hair': (255, 255, 255),
    'hat': (0, 0, 0),
    'ear_r': (255, 255, 255),
    'neck_l': (255, 255, 255),
    'neck': (255, 255, 255),
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

def apply_semantic_segmentation(model, dataset, save_folder='mask'):
    for i in range(len(dataset)):
        inputs = dataset[i]
        inputs = inputs.cuda()
        outputs = model(inputs.unsqueeze(0))
        _, pred = torch.max(outputs, 1)
        pred = pred.data.cpu().numpy().squeeze().astype(np.uint8)
        imname = os.path.basename(dataset.images[i])

        # 將灰度圖轉換為彩色圖像
        mask_color = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        for j, class_name in enumerate(dataset.CLASSES):
            mask_color[pred == j] = class_color_map[class_name]

        # 保存彩色語義分割結果
        mask_color_pred = Image.fromarray(mask_color)
        save_path = os.path.join(save_folder, imname[:-4] + '.jpg') #改
        mask_color_pred.save(save_path)

        print('processed {0}/{1} images'.format(i + 1, len(dataset)))


#Testing時候用的
def extract_faceandbackground(segmentation_folder, original_images_folder, face_output_folder, background_output_folder):

    segmentation_files = [f for f in os.listdir(segmentation_folder) if f.endswith('.jpg')]

    for segmentation_file in segmentation_files:
        # Form the full paths
        segmentation_path = os.path.join(segmentation_folder, segmentation_file)
        original_path = os.path.join(original_images_folder, segmentation_file)

        # Read segmentation result
        segmentation_result = Image.open(segmentation_path)
        # 將分割結果轉換為NumPy數組
        segmentation_array = np.array(segmentation_result)

        # 人在分割結果中的類別為白色（255, 255, 255），背景為黑色（0, 0, 0）
        face_mask = (segmentation_array >= [128, 128, 128]).all(axis=-1).astype(np.uint8)

        background_mask = (segmentation_array < [128, 128, 128]).all(axis=-1).astype(np.uint8)  # 背景的

        # Read original image
        original_image = Image.open(original_path)
        original_array = np.array(original_image)

        # 提取人臉區域
        face_region = original_array * np.expand_dims(face_mask, axis=2)
        # 背景區域
        background_region = original_array * np.expand_dims(1 - face_mask, axis=2)  # 改

        # 非人臉區域變成灰色
        non_face_color = [128, 128, 128]  # 灰色
        face_region = face_region + np.expand_dims(1 - face_mask, axis=2) * np.array(non_face_color)

        # Save the extracted face and background regions
        face_output_path = os.path.join(face_output_folder, segmentation_file)
        background_output_path = os.path.join(background_output_folder, segmentation_file)

        Image.fromarray(face_region.astype(np.uint8)).save(face_output_path)
        Image.fromarray(background_region).save(background_output_path)



#training時候用的
def extract_face_background(segmentation_folder, original_images_folder, output_folder, face_output_folder, background_output_folder):

    segmentation_files = [f for f in os.listdir(segmentation_folder) if f.endswith('.jpg')]

    for segmentation_file in segmentation_files:
        # Form the full paths
        segmentation_path = os.path.join(segmentation_folder, segmentation_file)
        original_path = os.path.join(original_images_folder, segmentation_file)

        # Read segmentation result
        segmentation_result = Image.open(segmentation_path)
        # 將分割結果轉換為NumPy數組
        segmentation_array = np.array(segmentation_result)

        # 人在分割結果中的類別為白色（255, 255, 255），背景為黑色（0, 0, 0）
        face_mask = (segmentation_array >= [128, 128, 128]).all(axis=-1).astype(np.uint8)

        background_mask = (segmentation_array < [128, 128, 128]).all(axis=-1).astype(np.uint8)  # 背景的

        # Read original image
        original_image = Image.open(original_path)
        original_array = np.array(original_image)

        # 提取人臉區域
        face_region = original_array * np.expand_dims(face_mask, axis=2)
        # 背景區域
        background_region = original_array * np.expand_dims(1 - face_mask, axis=2)  # 改

        # 非人臉區域變成灰色
        non_face_color = [128, 128, 128]  # 灰色
        face_region = face_region + np.expand_dims(1 - face_mask, axis=2) * np.array(non_face_color)


        # Save the extracted face and background regions
        face_output_path = os.path.join(face_output_folder, segmentation_file)
        background_output_path = os.path.join(background_output_folder, segmentation_file)

        Image.fromarray(face_region.astype(np.uint8)).save(face_output_path)
        Image.fromarray(background_region).save(background_output_path)

        # -------------以下是背景人臉結合
        # Convert masks to grayscale 使用貼上方法時，蒙版應為單通道影像，其值為 0（完全透明）到 255（完全不透明），face_mask和background_mask是三通道影像。要解決此問題，可以透過將遮罩轉換為灰階來將其轉換為單通道影像。
        #face_mask_gray = Image.fromarray(face_mask * 255)
        #background_mask_gray = Image.fromarray(background_mask * 255)

        # 將兩張圖片組合起來
        #output_image = Image.new("RGB", original_image.size)
        #output_image.paste(Image.fromarray(background_region), (0, 0), background_mask_gray)
        #output_image.paste(Image.fromarray(face_region), (0, 0), face_mask_gray)

        #output_path = os.path.join(output_folder, segmentation_file)
        # 保存組合後的圖片
        #output_image.save(output_path)


def combine_face_and_background(im_save_path, background_folder, segmentation_folder):
    # 確認 im_save_path 是否為 PNG 檔案
    assert im_save_path.endswith('.png'), "im_save_path 應該是 PNG 檔案"

    # 確保背景和遮罩資料夾存在
    assert os.path.isdir(background_folder), f"{background_folder} 不是一個資料夾"
    assert os.path.isdir(segmentation_folder), f"{segmentation_folder} 不是一個資料夾"

    # 讀取人臉圖片
    face_image = Image.open(im_save_path).convert("RGBA")  # 轉換為 RGBA 模式

    # 組合完整的遮罩檔案路徑
    mask_path = os.path.join(segmentation_folder, os.path.basename(im_save_path))

    # 讀取遮罩圖片
    mask = Image.open(mask_path).convert("L")  # 轉換為灰度模式


    #mask = mask.filter(ImageFilter.SMOOTH_MORE)


    # 將 mask 轉換為透明度通道
    alpha = mask.point(lambda p: p > 128 and 255)

    # 將 alpha 通道應用到人臉圖片
    face_image.putalpha(alpha)

    # 組合完整的背景檔案路徑
    background_path = os.path.join(background_folder, os.path.basename(im_save_path))

    # 檢查背景圖片是否存在
    if not os.path.exists(background_path):
        print(f"跳過 {os.path.basename(im_save_path)}，因為缺少背景圖片")
        return

    # 讀取背景圖片
    background_image = Image.open(background_path).convert("RGBA")  # 轉換為 RGBA 模式

    # 合併人臉和背景，這次將透明度通道應用到背景上
    result = Image.alpha_composite(background_image.copy(), face_image)

    # 檢查是否有完全黑色的像素
    #black_pixels = [(x, y) for x in range(result.width) for y in range(result.height) if result.getpixel((x, y))[:3] == (0, 0, 0)]

    # 如果有完全黑色的像素，將它們替換為最近的非黑色像素的顏色
    #for x, y in black_pixels:
    #    nearest_color = background_image.getpixel((x, y))
    #    result.putpixel((x, y), nearest_color[:3])  # 使用 [:3] 取得 RGB 顏色值

    # 檢查是否有完全黑色的像素
    black_pixels = [(x, y) for x in range(result.width) for y in range(result.height) if
                    result.getpixel((x, y))[:3] == (0, 0, 0)]

    # 如果有完全黑色的像素，將它們替換為最近的非黑色像素的顏色
    for x, y in black_pixels:
        nearest_color = background_image.getpixel((x, y))

        # 設定一個閾值，判斷是否處於黑色範圍附近
        threshold = 30  # 調整閾值的大小
        if all(abs(nearest_color[i] - (0, 0, 0)[i]) <= threshold for i in range(3)):
            result.putpixel((x, y), nearest_color[:3])  # 使用 [:3] 取得 RGB 顏色值

    # 保存合成後的圖片
    result.save(im_save_path)


def apply_semantic_segmentation_img(model, image_path): #單一圖像位址
    # 預處理圖片
    image = Image.open(image_path).convert('RGB')
    image = image.resize((256, 256), Image.BILINEAR)
    image = preprocess_image(image, flip=False, scale=None, crop=(256, 256))  # 假設 preprocess_image 是你的預處理函數

    # 將圖片轉換為模型輸入的格式
    inputs = transforms.ToTensor()(image).unsqueeze(0).cuda()

    # 進行語義分割
    outputs = model(inputs)
    _, pred = torch.max(outputs, 1)
    pred = pred.data.cpu().numpy().squeeze().astype(np.uint8)

    # 將灰度圖轉換為彩色圖像
    mask_color = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    for j, class_name in enumerate(dataset.CLASSES):
        mask_color[pred == j] = class_color_map[class_name]

    # 保存彩色語義分割結果
    mask_color_pred = Image.fromarray(mask_color)
    save_path = 'mask/' + os.path.basename(image_path)[:-4] + '.jpg'
    mask_color_pred.save(save_path)

    #print('Processed image:', os.path.basename(image_path))



def extract_face_background_img(image_path, output_folder):  #單一圖像位址
    # 讀取原始圖片
    original_image = Image.open(image_path)
    original_array = np.array(original_image)

    # 獲取語義分割結果的檔案路徑
    segmentation_file = 'mask' + os.path.basename(image_path)[:-4] + '.jpg'
    segmentation_path = os.path.join(output_folder, segmentation_file)

    # 讀取語義分割結果
    segmentation_result = Image.open(segmentation_path)
    segmentation_array = np.array(segmentation_result)

    # 人在分割結果中的類別為白色（255, 255, 255），背景為黑色（0, 0, 0）
    face_mask = (segmentation_array >= [128, 128, 128]).all(axis=-1).astype(np.uint8)
    background_mask = (segmentation_array < [128, 128, 128]).all(axis=-1).astype(np.uint8)

    # 提取人臉區域
    face_region = original_array * np.expand_dims(face_mask, axis=2)
    # 背景區域
    background_region = original_array * np.expand_dims(1 - face_mask, axis=2)

    # 儲存提取的人臉和背景區域
    face_output_path = os.path.join(output_folder, 'face' + os.path.basename(image_path))
    background_output_path = os.path.join(output_folder, 'background' + os.path.basename(image_path))

    Image.fromarray(face_region).save(face_output_path)
    Image.fromarray(background_region).save(background_output_path)

    print('Processed image:', os.path.basename(image_path))




def extract_face_background2(segmentation_folder, original_images_folder, output_folder, face_output_folder, background_output_folder):  #背景不是黑的
    segmentation_files = [f for f in os.listdir(segmentation_folder) if f.endswith('.jpg')]

    for segmentation_file in segmentation_files:
        # Form the full paths
        segmentation_path = os.path.join(segmentation_folder, segmentation_file)
        original_path = os.path.join(original_images_folder, segmentation_file)

        # Read segmentation result
        segmentation_result = Image.open(segmentation_path)
        segmentation_array = np.array(segmentation_result)

        # 人在分割結果中的類別為白色（255, 255, 255），背景為黑色（0, 0, 0）
        face_mask = (segmentation_array >= [128, 128, 128]).all(axis=-1).astype(np.uint8) * 255
        background_mask = (segmentation_array < [128, 128, 128]).all(axis=-1).astype(np.uint8)

        # Read original image
        original_image = Image.open(original_path)
        original_array = np.array(original_image)

        # 提取人臉區域
        face_region = original_array * np.expand_dims(face_mask, axis=2)
        face_image = Image.fromarray(face_region)

        # 將背景變成透明

        # 背景區域
        background_region = original_array * np.expand_dims(1 - face_mask, axis=2)
        background_image = Image.fromarray(background_region)


        # Save the extracted face and background regions
        face_output_path = os.path.join(face_output_folder, os.path.splitext(segmentation_file)[0] + '.png')
        background_output_path = os.path.join(background_output_folder, os.path.splitext(segmentation_file)[0] + '.png')

        face_image.save(face_output_path)
        background_image.save(background_output_path)
