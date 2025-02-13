import os
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import cv2
from PIL import Image
from FADING_util import util
from p2p import *
from null_inversion import *
from torchvision import transforms
import torch.nn as nn
import torchvision
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_images_tensor(directory, filename, age):
    images = {}
    # Construct the expected filename pattern
    pattern = f"{filename}_{age}.jpg"  # Adjust the file extension if needed
    # Full path to search for the image
    file_path = os.path.join(directory, pattern)
    # Check if the file exists
    if os.path.isfile(file_path):
        try:
            # Load image using PIL and convert to RGB
            img = Image.open(file_path).convert('RGB')
            images[pattern] = img

            # Get the last image (since we're only loading one, this is the same)
            new_img = images[pattern]

            # Convert image to NumPy array
            new_img_array = np.array(new_img)

            # Convert image to a tensor
            new_img_tensor = torch.from_numpy(new_img_array).permute(2, 0, 1).float() / 255.0

            # Normalize tensor to range [-1, 1]
            new_img_tensor = (new_img_tensor * 2) - 1

            # Flip the tensor horizontally 左右相反
            #new_img_tensor = torch.flip(new_img_tensor, dims=[2])

            # Return the processed tensor
            return new_img_tensor.to(device)

        except Exception as e:
            print(f"Error loading or processing image {file_path}: {e}")
    else:
        print(f"File not found: {file_path}")

    return None  # Return None if the image wasn't found or couldn't be processed


def load_images(directory, filename, age):
    try:
        pattern = f"{filename}_{age}.jpg"
        file_path = os.path.join(directory, pattern)
        if os.path.isfile(file_path):
            img = Image.open(file_path).convert('RGB')
            return img
        else:
            print(f"File not found: {file_path}")
            return None
    except Exception as e:
        print(f"Error loading or processing image {file_path}: {e}")
        return None


def age_editing_pipeline(image_tensor, age_init, gender, specialized_path, save_aged_dir, target_ages):
    if not os.path.exists(save_aged_dir):
        os.makedirs(save_aged_dir)

    gt_gender = int(gender == 'female')
    person_placeholder = util.get_person_placeholder(age_init, gt_gender)
    inversion_prompt = f"photo of {age_init} year old {person_placeholder}"

    # 不再使用路徑，而是使用影像張量
    input_img_name = "aged_image"

    # 將tensor的值從 [-1, 1] 歸一化到 [0, 1]
    image_tensor = (image_tensor + 1) / 2


    # 水平方向翻轉tensor
    image_tensor = torch.flip(image_tensor, dims=[2])

    #print('image_tensor:', image_tensor.shape #image_tensor: torch.Size([3, 256, 256])
    image_tensor = image_tensor.permute(1, 2, 0)  # 把原本C*H*W [3, 256, 256]換成 H*W*C [256, 256, 3]
    #print('image_tensor:', image_tensor.shape)  #   image_tensor: torch.Size([256, 256, 3])


    # 將 Tensor 轉換為 PIL 影像
    # to_pil = transforms.ToPILImage()
    # image_pil = to_pil(image_tensor.cpu())  # 將影像張量轉換為 PIL 影像，並確保它在 CPU 上
    #image_pil = transforms.ToPILImage()(image_tensor.cpu())
    #image_pil = Image.fromarray((image_tensor.cpu().numpy() * 255).astype('uint8'))

    # 直接將張量轉換為 NumPy 陣列
    image_np = (image_tensor.cpu().numpy() * 255).astype('uint8')

    # 顯示 NumPy 陣列轉換的圖像（如果需要）
    #image_pil = Image.fromarray(image_np)
    #image_pil.show(title="Initial Image (PIL)")



    # 加载 specialized diffusion model
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                              clip_sample=False, set_alpha_to_one=False,
                              steps_offset=1)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    #torch.cuda.set_per_process_memory_fraction(0.2, 0)

    g_cuda = torch.Generator(device=device)

    ldm_stable = StableDiffusionPipeline.from_pretrained(specialized_path,
        scheduler=scheduler,
        safety_checker=None).to(device)
    tokenizer = ldm_stable.tokenizer

    # Null text inversion
    null_inversion = NullInversion(ldm_stable)
    (image_gt, image_enc), x_t, uncond_embeddings = null_inversion.invert(image_np, inversion_prompt,
                                                                          offsets=(0,0,0,0), verbose=True)

    # 定義自適應池化層來進行resize , psp.py裡的face_pool
    adaptive_pool = nn.AdaptiveAvgPool2d((256, 256))

    # 年龄编辑
    for age_new in target_ages:
        print(f'Age editing with target age {age_new}...')
        new_person_placeholder = util.get_person_placeholder(age_new, gt_gender)
        new_prompt = inversion_prompt.replace(person_placeholder, new_person_placeholder)
        new_prompt = new_prompt.replace(str(age_init), str(age_new))
        blend_word = (((str(age_init), person_placeholder), (str(age_new), new_person_placeholder)))
        is_replace_controller = True

        prompts = [inversion_prompt, new_prompt]

        cross_replace_steps = {'default_': .8}
        self_replace_steps = .5

        eq_params = {"words": (str(age_new)), "values": (1,)}

        controller = make_controller(prompts, is_replace_controller, cross_replace_steps, self_replace_steps,
                                     tokenizer, blend_word, eq_params)

        images, _ = p2p_text2image(ldm_stable, prompts, controller, generator=g_cuda.manual_seed(0),
                                   latent=x_t, uncond_embeddings=uncond_embeddings)

        new_img = images[-1]
        new_img_resized = cv2.resize(new_img, (256, 256), interpolation=cv2.INTER_AREA)
        new_img_pil = Image.fromarray(new_img_resized)



        #new_img_pil.show(title="Generated Image (After p2p_text2image)")


        # 返回處理後的圖像
        new_img_pil.save(os.path.join(save_aged_dir, f'{input_img_name}_{age_new}.jpg'))


        #
        new_img_tensor = torch.from_numpy(new_img_resized).permute(2, 0, 1).float() / 255.0
        # 将张量的值从 [0, 1] 归一化回 [-1, 1]
        new_img_tensor = (new_img_tensor * 2) - 1
        # 水平方向翻转回去
        new_img_tensor = torch.flip(new_img_tensor, dims=[2])



    return new_img_tensor



#--------------------------- 未修改的
# def age_editing_pipeline(image_path, age_init, gender, specialized_path, save_aged_dir, target_ages):
#     if not os.path.exists(save_aged_dir):
#         os.makedirs(save_aged_dir)
#
#     gt_gender = int(gender == 'female')
#     person_placeholder = util.get_person_placeholder(age_init, gt_gender)
#     inversion_prompt = f"photo of {age_init} year old {person_placeholder}"
#
#     input_img_name = image_path.split('/')[-1].split('.')[-2]
#
#     # Load specialized diffusion model
#     scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
#                               clip_sample=False, set_alpha_to_one=False,
#                               steps_offset=1)
#     device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
#     g_cuda = torch.Generator(device=device)
#
#     ldm_stable = StableDiffusionPipeline.from_pretrained(specialized_path,
#         scheduler=scheduler,
#         safety_checker=None).to(device)
#     tokenizer = ldm_stable.tokenizer
#
#     # Null text inversion
#     null_inversion = NullInversion(ldm_stable)
#     (image_gt, image_enc), x_t, uncond_embeddings = null_inversion.invert(image_path, inversion_prompt,
#                                                                           offsets=(0,0,0,0), verbose=True)
#
#     # Age editing
#     for age_new in target_ages:
#         print(f'Age editing with target age {age_new}...')
#         new_person_placeholder = util.get_person_placeholder(age_new, gt_gender)
#         new_prompt = inversion_prompt.replace(person_placeholder, new_person_placeholder)
#         new_prompt = new_prompt.replace(str(age_init), str(age_new))
#         blend_word = (((str(age_init), person_placeholder), (str(age_new), new_person_placeholder)))
#         is_replace_controller = True
#
#         prompts = [inversion_prompt, new_prompt]
#
#         cross_replace_steps = {'default_': .8}
#         self_replace_steps = .5
#
#         eq_params = {"words": (str(age_new)), "values": (1,)}
#
#         controller = make_controller(prompts, is_replace_controller, cross_replace_steps, self_replace_steps,
#                                      tokenizer, blend_word, eq_params)
#
#         images, _ = p2p_text2image(ldm_stable, prompts, controller, generator=g_cuda.manual_seed(0),
#                                    latent=x_t, uncond_embeddings=uncond_embeddings)
#
#         new_img = images[-1]
#         new_img_resized = cv2.resize(new_img, (256, 256), interpolation=cv2.INTER_AREA)
#         new_img_pil = Image.fromarray(new_img_resized)
#
#         #return new_img_pil
#         new_img_pil.save(os.path.join(save_aged_dir, f'{input_img_name}_{age_new}.jpg'))


#python age_editing.py --image_path D:/FADING-master/00204.jpg --age_init 20 --gender male --save_aged_dir D:/FADING-master/output --specialized_path D:/SAM-master/finetune_double_prompt_150_random --target_ages 10 20 80

#age_editing_pipeline(image_path='D:/FADING-master/00204.jpg',25, gender='male', specialized_path='D:/SAM-master/finetune_double_prompt_150_random', save_aged_dir='D:/FADING-master/output' ,target_ages=[80])
#save_aged_dir = 'D:/FADING-master/output'
#x.save(os.path.join(save_aged_dir, f'{00204}_{target_ages}.jpg'))
# age_editing_pipeline(
#     image_path='D:/FADING-master/00204.jpg',
#     age_init=25,
#     gender='male',
#     specialized_path='D:/SAM-master/finetune_double_prompt_150_random',
#     save_aged_dir='D:/FADING-master/o3',
#     target_ages=[80]
# )