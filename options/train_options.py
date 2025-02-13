from argparse import ArgumentParser
from configs.paths_config import model_paths  #所有的model_path都在configs/paths_config.py改


class TrainOptions:

    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        self.parser.add_argument('--exp_dir', type=str,
                                 help='Path to experiment output directory')
        self.parser.add_argument('--dataset_type', default='ffhq_aging', type=str,
                                 help='Type of dataset/experiment to run')
        self.parser.add_argument('--input_nc', default=4, type=int,
                                 help='Number of input image channels to the psp encoder')
        self.parser.add_argument('--label_nc', default=0, type=int,
                                 help='Number of input label channels to the psp encoder')
        self.parser.add_argument('--output_size', default=1024, type=int,    # output_size用在/home/tony/AGE-master/models/psp.py
                                 help='Output size of generator')

        self.parser.add_argument('--batch_size', default=4, type=int,
                                 help='Batch size for training')
        self.parser.add_argument('--test_batch_size', default=2, type=int,
                                 help='Batch size for testing and inference')
        self.parser.add_argument('--workers', default=4, type=int,
                                 help='Number of train dataloader workers')
        self.parser.add_argument('--test_workers', default=2, type=int,
                                 help='Number of test/inference dataloader workers')

        self.parser.add_argument('--learning_rate', default=0.0001, type=float,
                                 help='Optimizer learning rate')
        self.parser.add_argument('--optim_name', default='ranger', type=str,
                                 help='Which optimizer to use')  #ranger 換成 adam
        self.parser.add_argument('--train_decoder', action='store_true',
                                 help='Whether to train the decoder model')
        self.parser.add_argument('--start_from_latent_avg', action='store_true',
                                 help='Whether to add average latent vector to generate codes from encoder.')
        self.parser.add_argument('--start_from_encoded_w_plus', action='store_true',
                                 help='Whether to learn residual wrt w+ of encoded image using pretrained pSp.')

        self.parser.add_argument('--lpips_lambda', default=0, type=float,
                                 help='LPIPS loss multiplier factor')
        self.parser.add_argument('--id_lambda', default=0, type=float,
                                 help='ID loss multiplier factor')
        self.parser.add_argument('--l2_lambda', default=0, type=float,
                                 help='L2 loss multiplier factor')
        self.parser.add_argument('--w_norm_lambda', default=0, type=float,
                                 help='W-norm loss multiplier factor')
        self.parser.add_argument('--aging_lambda', default=0, type=float,
                                 help='Aging loss multiplier factor')
        self.parser.add_argument('--cycle_lambda', default=0, type=float,
                                 help='Cycle loss multiplier factor')

        self.parser.add_argument('--lpips_lambda_crop', default=0, type=float,
                                 help='LPIPS loss multiplier factor for inner image region')
        self.parser.add_argument('--l2_lambda_crop', default=0, type=float,
                                 help='L2 loss multiplier factor for inner image region')

        self.parser.add_argument('--lpips_lambda_aging', default=0, type=float,
                                 help='LPIPS loss multiplier factor for aging')
        self.parser.add_argument('--l2_lambda_aging', default=0, type=float,
                                 help='L2 loss multiplier factor for aging')

        self.parser.add_argument('--stylegan_weights', default=model_paths['stylegan_ffhq'], type=str,#所有的model_path都在configs/paths_config.py改
                                 help='Path to StyleGAN model weights')
        self.parser.add_argument('--checkpoint_path', default=None, type=str,
                                 help='Path to pSp model checkpoint')

        self.parser.add_argument('--max_steps', default=500000, type=int,
                                 help='Maximum number of training steps')
        self.parser.add_argument('--image_interval', default=100, type=int,
                                 help='Interval for logging train images during training')
        self.parser.add_argument('--board_interval', default=50, type=int,
                                 help='Interval for logging metrics to tensorboard')
        self.parser.add_argument('--val_interval', default=1000, type=int,
                                 help='Validation interval')
        self.parser.add_argument('--save_interval', default=None, type=int,
                                 help='Model checkpoint interval')

        # arguments for aging
        self.parser.add_argument('--target_age', default=None, type=str,
                                 help='Target age for training. Use `uniform_random` for random sampling of target age')
        self.parser.add_argument('--use_weighted_id_loss', action="store_true",
                                 help="Whether to weight id loss based on change in age (more change -> less weight)")
        self.parser.add_argument('--pretrained_psp_path', default=model_paths['pretrained_psp_encoder'], type=str,
                                 help="Path to pretrained pSp network.") #所有的model_path都在configs/paths_config.py改

        #discriminator
        self.parser.add_argument('--gan_loss_weight', type=float, help='gan_loss_weight', default=30)  # 新的鑑別器 IPCGAN是75(用過10了,20)
        self.parser.add_argument('--d_learning_rate', '--lr', type=float, help='discriminator learning rate', default=1e-4)

        #shape_loss
        self.parser.add_argument('--shape_lambda', default=1, type=float,
                                 help='shape loss multiplier factor') #2
        self.parser.add_argument('--penalty_value', default=1.2, type=float,
                                 help='penalty value of shape loss')
        self.parser.add_argument('--face_landmarks_path', default=model_paths['68_face_landmarks'], type=str,
                                 help='Path to 68_face_landmarks')

        self.parser.add_argument('--fading_lambda', default=0, type=float,
                                 help='FADING loss')  #1  3


        #hair_color_loss
        self.parser.add_argument('--color_lambda', default=0, type=float,
                                 help='color loss multiplier factor') #5





    def parse(self):
        opts = self.parser.parse_args()
        return opts
