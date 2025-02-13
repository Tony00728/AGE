import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Conv2d, BatchNorm2d, PReLU, Sequential, Module


from models.encoders.helpers import get_blocks, bottleneck_IR, bottleneck_IR_SE
from models.stylegan2.model import EqualLinear
from models.encoders.CBAM import CBAM, SpatialAttentionModule, ChannelAttentionModule,Newmodule

class GradualStyleBlock(Module):
    def __init__(self, in_c, out_c, spatial):
        super(GradualStyleBlock, self).__init__()
        self.out_c = out_c
        self.spatial = spatial
        num_pools = int(np.log2(spatial))
        modules = []
        modules += [Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1), nn.LeakyReLU()]
        for i in range(num_pools - 1):
            modules += [
                Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1), nn.LeakyReLU()
            ]
        self.convs = nn.Sequential(*modules)
        self.linear = EqualLinear(out_c, out_c, lr_mul=1)



    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.out_c)  #等同於reshape -1表示一個不確定的數（列）但是很肯定要成self.out_c行  ,self.out_c是行
        x = self.linear(x)
        return x


class new_GradualStyleEncoder(Module):   #這邊都用Module 前面有from torch.nn了
    def __init__(self, num_layers, mode='ir', n_styles=18, opts=None):
        super(new_GradualStyleEncoder, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'   # num_layers只有50、100、152
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'        # mode只有ir或ir_se
        blocks = get_blocks(num_layers)     # 在/home/tony/SAM-master/models/encoders/helpers.py
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE  # models/psp.py 現在是用ir_se
        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))  #GradualStyleEncoder的input layer
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)  #創建了 body，它是一個包含了所有modules網絡主體部分模塊的 Sequential 串聯模塊。


        self.styles = nn.ModuleList()
        self.style_count = n_styles  #總共有18個n_styles
        self.coarse_ind = 3  #psp 模型藉由 FPN 架構，將不同 Scale 的特徵輸入至對應 Scale 的 StyleGAN，並將 18 個 Style code(W+）分為三個部分，依序為 Coarse, Medium and Fine 對應至 StyleGAN 中不同 Scale 的位置。
        self.middle_ind = 7
        for i in range(self.style_count):
            if i < self.coarse_ind:  # 0,1,2  3個
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind: #3到6  4個
                style = GradualStyleBlock(512, 512, 32)
            else:  #7到17  11個
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)
        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

        self.Newmodule_c1 = Newmodule(channel=128)
        self.Newmodule_c2 = Newmodule(channel=256)
        self.Newmodule_c3 = Newmodule(channel=512)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
		Args:
		  x: (Variable) top feature map to be upsampled.
		  y: (Variable) lateral feature map.
		Returns:
		  (Variable) added feature map.
		Note in PyTorch, when input size is odd, the upsampled feature map
		with `F.upsample(..., scale_factor=2, mode='nearest')`
		maybe not equal to the lateral feature map size.
		e.g.
		original input size: [N,_,15,15] ->
		conv2d feature map size: [N,_,8,8] ->
		upsampled feature map size: [N,_,16,16]
		So we choose bilinear upsample which supports arbitrary output sizes.
		'''
        _, _, H, W = y.size()  # size : N,C,H,W
        #這一行程式碼進行了上取樣操作，將頂部特徵圖x上取樣到與側面特徵圖y相同的大小，並使用雙線性插值方法。 然後，它將上取樣後的特徵圖與側面特徵圖相加，得到最終的特徵圖，這就是函數的回傳值。 這個操作用於將不同尺寸的特徵圖進行融合。
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x):

        x = self.input_layer(x)

        latents = []
        modulelist = list(self.body._modules.values())  #將模型的body部分的所有子模組（modules）提取並轉換為一個列表modulelist
        for i, l in enumerate(modulelist): #遍歷modulelist中的所有子模組，同時使用i來追蹤目前子模組的索引，l代表目前的子模組。 (body的架構有0到23層）
            x = l(x)
            if i == 6:
                c1 = x
                #print(" c1 shape= ", c1.shape)
                c1 = self.Newmodule_c1(c1)

            elif i == 20:
                c2 = x
                #print(" c2 shape= ", c2.shape)
                c2 = self.Newmodule_c2(c2)


            elif i == 23:  #body的最後一層
                c3 = x
                #print(" c3 shape= ", c3.shape)
                c3 = self.Newmodule_c3(c3)

        for j in range(self.coarse_ind):  #coarse_ind = 3
            latents.append(self.styles[j](c3))

        p2 = self._upsample_add(c3, self.latlayer1(c2))

        for j in range(self.coarse_ind, self.middle_ind):  #coarse_ind = 3  ,middle_ind = 7
            latents.append(self.styles[j](p2))

        p1 = self._upsample_add(p2, self.latlayer2(c1))

        for j in range(self.middle_ind, self.style_count):
            latents.append(self.styles[j](p1))

        out = torch.stack(latents, dim=1)
        return out



''''
get_blocks(50)
b = GradualStyleEncoder(50, 'ir_se', 18 )
print(b)
'''