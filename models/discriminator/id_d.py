import torch.nn as nn
import torch
import torch.nn.functional as F
import os
from PIL import Image
import torchvision.transforms as transforms

from utils.network import Conv2d  #same padding  輸出大小為原來大小/stride


class PatchDiscriminator(nn.Module):
    def __init__(self):
        super(PatchDiscriminator, self).__init__()
        self.lrelu = nn.LeakyReLU(0.2)
        self.conv1 = Conv2d(3, 64, kernel_size=4, stride=2)
        self.conv2 = Conv2d(69, 128, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(128, eps=0.001, track_running_stats=True)
        self.conv3 = Conv2d(128, 256, kernel_size=4, stride=2)
        self.bn3 = nn.BatchNorm2d(256, eps=0.001, track_running_stats=True)
        self.conv4 = Conv2d(256, 512, kernel_size=4, stride=2)
        self.bn4 = nn.BatchNorm2d(512, eps=0.001, track_running_stats=True)
        self.conv5 = Conv2d(512, 512, kernel_size=4, stride=2)

    def forward(self, x,condition):
        x = self.lrelu(self.conv1(x))
        x=torch.cat((x,condition),1)   #x後面接著condition ,1為按著行拼接
        x = self.lrelu(self.bn2(self.conv2(x)))
        x = self.lrelu(self.bn3(self.conv3(x)))
        x = self.lrelu(self.bn4(self.conv4(x)))
        x = self.conv5(x)
        return x


class ConditionalDiscriminator(nn.Module):
    def __init__(self, img_channels=3):
        super(ConditionalDiscriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(img_channels * 2, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()  # 輸出0到1之間的值
        )

    def forward(self, generated_image, original_input_image):
        combined = torch.cat((generated_image, original_input_image), dim=1)
        validity = self.model(combined)
        return validity.view(-1, 1).squeeze(1)  #變為1維


def Conditionaldiscriminator_loss(discriminator, real_images, generated_images):
    real_validity = discriminator(real_images, real_images)  # 使用原始影像作為條件比較真實性
    fake_validity = discriminator(generated_images.detach(), real_images)  # 分離生成影像，避免梯度傳播到生成器

    real_loss = F.binary_cross_entropy(real_validity, torch.ones_like(real_validity))
    fake_loss = F.binary_cross_entropy(fake_validity, torch.zeros_like(fake_validity))

    d_loss = (real_loss + fake_loss) / 2  # 取兩者損失的平均值作為鑑別器的損失
    return d_loss


#####
# if __name__=="__main__":
#     tensor = torch.ones((1, 3 , 1024, 1024))
#     tensor2 = torch.ones((1, 3, 1024, 1024))
#     image_path = '/home/tony/FFHQ--Aging/test/20-29/56655.jpg'
#     image_path2 = '/home/tony/FFHQ_test_pretrain/20-29/inference_results/20/56655.jpg'
#
#     discriminator = ConditionalDiscriminator()
#     image = Image.open(image_path).convert('RGB')  # 確保圖片是 RGB 格式
#     image2 = Image.open(image_path2).convert('RGB')  # 確保圖片是 RGB 格式
#
#     # 定義圖像轉換
#     transform = transforms.Compose([
#         transforms.Resize((128, 128)),  # 調整圖片大小
#         transforms.ToTensor(),  # 轉換成張量
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正規化
#     ])
#
#     # 對圖片應用轉換
#     image_tensor = transform(image).unsqueeze(0)  # 添加批次維度
#     image_tensor2 = transform(image2).unsqueeze(0)  # 添加批次維度
#     # 將處理後的圖片張量輸入到 discriminator
#     output2 = discriminator(image_tensor, image_tensor2)  # 假設 discriminator 是可接受張量作為輸入的模型
#     print(output2)
#     print(output2.size())
#
#     dloss = Conditionaldiscriminator_loss(discriminator,image_tensor,image_tensor2)
#     print(dloss)