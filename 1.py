
import torch
from torch import nn
import math
from models.encoders.helpers import get_blocks, bottleneck_IR, bottleneck_IR_SE
from models.stylegan2.model import EqualLinear
from configs.paths_config import model_paths
from models.encoders import psp_encoders
from models.stylegan2.model import Generator
# print(torch.__version__)
# print(torch.version.cuda)
# print(torch.cuda.is_available())

# a = torch.arange(1,17)
# print(a.view(-1,2))

import torchvision.models as models
from torchinfo import summary

get_blocks(50)
b =  psp_encoders.GradualStyleEncoder(50, 'ir_se', 18 )
summary(b, (1, 4, 256, 256))
