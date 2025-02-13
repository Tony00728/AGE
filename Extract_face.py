from Extract import *
import os

if __name__ == "__main__":
    # 下載模型
    download_models()

    # 載入模型
    resolution = 256
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    model_fname = 'deeplab_model/deeplab_model.pth'
    dataset_root = 'D:/FADING-master/ablation'
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

    # 應用語義分割
    apply_semantic_segmentation(model, dataset, save_folder='D:/FADING-master/segmentation/mask')



    segmentation_folder = 'D:/FADING-master/segmentation/mask'
    original_images_folder = 'D:/FADING-master/ablation/image'
    output_folder = 'D:/FADING-master/segmentation/combined_image'
    face_output_folder = 'C:/paper_img/img/ablation/img'
    background_output_folder = 'D:/FADING-master/segmentation/background'

    if not os.path.exists(face_output_folder):
        os.makedirs(face_output_folder)
    if not os.path.exists(background_output_folder):
        os.makedirs(background_output_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # 提取人臉和背景
    extract_face_background(segmentation_folder, original_images_folder, output_folder, face_output_folder, background_output_folder) #結合先不用做