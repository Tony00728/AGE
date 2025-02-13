#在這邊做語義分割
from Extract import *
import os

# 檢查標誌檔案是否存在
flag_file_path = 'D:/AGE-master/semantic_segmentation_done.txt'
semantic_segmentation_done = os.path.exists(flag_file_path)

if not semantic_segmentation_done:

    # 下載模型
    download_models()

    # 載入模型
    resolution = 256
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    model_fname = 'deeplab_model/deeplab_model.pth'
    #train
    dataset_root = 'D:/AGE-master/ffhq_white_females' #裡面在一個資料夾 再放圖片  D:/AGE-master/morph_train
    assert os.path.isdir(dataset_root)
    dataset = CelebASegmentation(dataset_root, crop_size=256)
    #val
    val_dataset_root = 'D:/AGE-master/ffhq_white_females_val' #裡面在一個資料夾 再放圖片
    assert os.path.isdir(val_dataset_root)
    val_dataset = CelebASegmentation(val_dataset_root, crop_size=256)

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

    # 應用語義分割
    apply_semantic_segmentation(model, dataset, save_folder='D:/AGE-master/mask')

    #val
    model2 = getattr(deeplab, 'resnet101')(
        pretrained=True,
        num_classes=len(val_dataset.CLASSES),
        num_groups=32,
        weight_std=True,
        beta=False)

    model2 = model2.cuda()
    model2.eval()
    checkpoint = torch.load(model_fname)
    state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k}
    model2.load_state_dict(state_dict)

    # 應用語義分割
    apply_semantic_segmentation(model2, val_dataset, save_folder='D:/AGE-master/val_mask')


    #train
    segmentation_folder = 'D:/AGE-master/mask'
    original_images_folder = 'D:/AGE-master/ffhq_white_females/image'  # 改圖片資料夾
    output_folder = 'D:/AGE-master/train_face/combined'
    face_output_folder = 'D:/AGE-master/train_face/face'
    background_output_folder = 'D:/AGE-master/train_face/background'

    #val
    val_segmentation_folder = 'D:/AGE-master/val_mask'
    val_images_folder = 'D:/AGE-master/ffhq_white_females_val/image' #改圖片資料夾
    val_output_folder = 'D:/AGE-master/val_face/combined'
    val_face_output_folder = 'D:/AGE-master/val_face/face'
    val_background_output_folder = 'D:/AGE-master/val_face/background'

    #train
    if not os.path.exists(face_output_folder):
        os.makedirs(face_output_folder)
    if not os.path.exists(background_output_folder):
        os.makedirs(background_output_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # 提取人臉和背景
    extract_face_background(segmentation_folder, original_images_folder, output_folder, face_output_folder, background_output_folder) #結合先不用做


    #val
    if not os.path.exists(val_face_output_folder):
        os.makedirs(val_face_output_folder)
    if not os.path.exists(val_background_output_folder):
        os.makedirs(val_background_output_folder)
    if not os.path.exists(val_output_folder):
        os.makedirs(val_output_folder)
    # 提取人臉和背景
    extract_face_background(val_segmentation_folder, val_images_folder, val_output_folder, val_face_output_folder, val_background_output_folder) #結合先不用做
    # 將標誌檔案寫入
    with open(flag_file_path, 'w') as flag_file:
        flag_file.write('Done')
    print("Semantic segmentation has been executed.")
else:
    print("Semantic segmentation has already been executed.")


dataset_paths = {
    'celeba_test': 'D:/AGE-master/CelebAMask-HQ/test_img',
    'ffhq': 'D:/AGE-master/train_face/face',   #   face_output_folder  , 'D:/AGE-master/train_face/face'直接打face_output_folder的資料夾
    'ffhq_val':  'D:/AGE-master/val_face/face', #   val_face_output_folder,   'D:/AGE-master/val_face/face'
    'morph': 'D:/AGE-master/train_face/face',
    'morph_val': 'D:/AGE-master/val_face/face',
}

model_paths = {
    'pretrained_psp_encoder': 'pretrained_models/psp_ffhq_encode.pt',
    'ir_se50': 'pretrained_models/model_ir_se50.pth',
    'stylegan_ffhq': 'pretrained_models/stylegan2-ffhq-config-f.pt',  #stylegan3-t-ffhq-1024x1024.pkl
    'shape_predictor': 'shape_predictor_68_face_landmarks.dat',
    'age_predictor': 'pretrained_models/dex_age_classifier.pth',
    '68_face_landmarks': 'D:/AGE-master/shape_predictor_68_face_landmarks.dat'
}
