import os
import argparse
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch
import cv2
from PIL import Image  # Ensure you have imported this for saving images

from FADING_util import util
from p2p import *
from null_inversion import *
from criteria.aging_loss import AgingLoss  # Make sure this is the correct path to AgingLoss

# Initialize the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--image_folder', required=True, help='Path to input image folder')
parser.add_argument('--age_init', type=str, help='Initial age will be calculated automatically if not specified')
parser.add_argument('--gender', required=True, choices=["female", "male"],
                    help="Specify the gender ('female' or 'male')")
parser.add_argument('--specialized_path', required=True, help='Path to specialized diffusion model')
parser.add_argument('--save_aged_dir', default='./outputs', help='Path to save outputs')
parser.add_argument('--target_ages', nargs='+', default=[10, 20, 40, 60, 80], type=int, help='Target age values')

args = parser.parse_args()

# Extract arguments
image_folder = args.image_folder
age_init = None if args.age_init == 'None' else int(args.age_init)

gender = args.gender
save_aged_dir = args.save_aged_dir
specialized_path = args.specialized_path
target_ages = args.target_ages

# Create the save directory if it does not exist
if not os.path.exists(save_aged_dir):
    os.makedirs(save_aged_dir)

# Determine device
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
g_cuda = torch.Generator(device=device)

# Initialize the aging loss to get the initial age

aging_loss = AgingLoss(args)

# Process each image in the folder
for image_name in os.listdir(image_folder):
    if image_name.endswith('.png') or image_name.endswith('.jpg'):
        image_path = os.path.join(image_folder, image_name)
        image = cv2.imread(image_path)
        image_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float().to(device)  # Convert to tensor


        # Use a local variable to store the initial age for each image
        local_age_init = None

        # Measure the initial age if not specified or recalculate for each image
        if age_init is None:
            age_init_tensor = aging_loss.extract_ages(image_tensor)
            local_age_init = int(age_init_tensor.item())  # Get the initial age as an integer
        else:
            local_age_init = age_init  # Use the global age_init if it's specified


        print(f'Image:{image_path},Initial age detected:{local_age_init}')

        gt_gender = int(gender == 'female')
        person_placeholder = util.get_person_placeholder(local_age_init, gt_gender)
        inversion_prompt = f"photo of {local_age_init} year old {person_placeholder}"

        # Load specialized diffusion model
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                                  set_alpha_to_one=False, steps_offset=1)
        ldm_stable = StableDiffusionPipeline.from_pretrained(specialized_path, scheduler=scheduler,
                                                             safety_checker=None).to(device)
        tokenizer = ldm_stable.tokenizer

        # Null text inversion
        null_inversion = NullInversion(ldm_stable)
        (image_gt, image_enc), x_t, uncond_embeddings = null_inversion.invert(image_path, inversion_prompt,
                                                                              offsets=(0, 0, 0, 0), verbose=True)

        # Age editing loop
        input_img_name = image_name.split('.')[0]
        for age_new in target_ages:
            print(f'Age editing with target age {age_new}...')
            new_person_placeholder = util.get_person_placeholder(age_new, gt_gender)
            new_prompt = inversion_prompt.replace(person_placeholder, new_person_placeholder)
            new_prompt = new_prompt.replace(str(local_age_init), str(age_new))
            blend_word = (((str(local_age_init), person_placeholder,), (str(age_new), new_person_placeholder,)))
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
            new_img_pil.save(os.path.join(save_aged_dir, f'{input_img_name}_{age_new}.jpg'))
