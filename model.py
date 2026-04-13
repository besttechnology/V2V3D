import torch.nn as nn
import torch
from torch.nn import functional as F
from torch.fft import fft2, ifft2
from utils import warp_feats
import numpy as np

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class Upscale(nn.Module):
    def __init__(self,in_channels,out_channels,mode,scale=None):
        super().__init__()
        upscale = []
        if mode == 'subpixel':
            upscale.append(nn.PixelShuffle(scale))
            upscale.append(nn.Conv2d(in_channels=int(in_channels/(scale**2)),out_channels=out_channels,kernel_size=3,stride=1,padding=1))
            self.upscale = nn.Sequential(*upscale)

        if mode == 'upconv':
            upscale.append(nn.UpsamplingNearest2d(scale_factor=scale))
            upscale.append(nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1))
            self.upscale = nn.Sequential(*upscale)
    def forward(self,x):
        x = self.upscale(x)
        return x


class Encoder(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        conv = []
        conv.append(nn.LeakyReLU(0.01))
        conv.append(nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1))

        self.conv = nn.Sequential(*conv)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)

    def forward(self,x):
        x = self.conv(x)
        x = self.pool(x)

        return x

class Decoder(nn.Module):
    def __init__(self,in_channels,out_channels,scale):
        super().__init__()
        decoder = []
        decoder.append(nn.LeakyReLU(0.01))
        decoder.append(Upscale(in_channels=in_channels,out_channels=out_channels,scale=scale,mode='upconv'))
        self.decoder = nn.Sequential(*decoder)

    def forward(self,x):
        x = self.decoder(x)
        return x

class conv_2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(conv_2d, self).__init__()
        pad = kernel_size // 2 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad)

    def forward(self, x):
        return self.conv(x) 

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,  dilation,  downsample=None):
        super(conv_block, self).__init__()
        
        self.downsample = downsample
        block = nn.ModuleList()
        block.append(nn.Conv2d(in_channels, out_channels, 3,1,1, dilation=dilation))
        block.append(nn.LeakyReLU(0.01, True))
        block.append(nn.Conv2d(out_channels, out_channels, 3,1,1, dilation=dilation))
        self.conv = nn.Sequential(*block)

    def forward(self, x):
        x_skip = x
        if self.downsample is not None:
            x_skip = self.downsample(x)
        return self.conv(x) + x_skip

class Feature(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Feature, self).__init__()
        self.relu = nn.LeakyReLU(0.01, True)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 4, 3, 1,1), nn.LeakyReLU(0.01, True))
        
        self.layer1 = self._make_layer(4, 4, 2, 1, 1)
        self.layer2 = self._make_layer(4, 8, 2, 1, 1)
        self.layer3 = self._make_layer(8, 16, 2, 1, 1)
       
        self.branch1 = nn.Sequential(nn.AvgPool2d(2,2), conv_2d(16, 4,1,1), nn.LeakyReLU(0.01, True), nn.UpsamplingBilinear2d(scale_factor=2))
        self.branch2 = nn.Sequential(nn.AvgPool2d(4,4), conv_2d(16, 4,1,1), nn.LeakyReLU(0.01, True), nn.UpsamplingBilinear2d(scale_factor=4))
        self.branch3 = nn.Sequential(nn.AvgPool2d(8,8), conv_2d(16, 4,1,1), nn.LeakyReLU(0.01, True), nn.UpsamplingBilinear2d(scale_factor=8))
       
        self.lastconv = nn.Sequential(conv_2d(28, 16, 3, 1), nn.LeakyReLU(0.01, True), nn.Conv2d(16,out_channels,1,1),nn.LeakyReLU(0.01, True))

    def _make_layer(self, in_c, out_c, blocks, stride, dilation):
        downsample = None
        if stride != 1 or in_c != out_c:
            downsample = conv_2d(in_c, out_c,1,1) 
        
        layers = []
        layers.append(conv_block(in_c, out_c, 3, stride, dilation, downsample))
        for _ in range(1, blocks):
            layers.append(conv_block(out_c, out_c, 3, stride, dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        
        x = torch.cat([l3, self.branch1(l3), self.branch2(l3), self.branch3(l3)], 1)
        x = self.lastconv(x)
        return x


class GaussianFeatureAlignment(nn.Module):
    """基于 Gaussian 软核的可学习特征对齐模块。

    替代原始的 0/1 质心 mask + FFT 卷积对齐。
    - 使用 PSF 先验生成的 Gaussian 软核做基础对齐
    - 附加轻量预测头，对每个视图的特征预测逐深度的权重调整
    - 残差设计：初始行为等价于使用 Gaussian 软核（无学习残差时）
    """

    def __init__(self, gauss_warp_psfs, n_views, n_depths, feat_ch):
        """
        Args:
            gauss_warp_psfs: (U, Z, H_k, W_k) Gaussian 软卷积核（已裁剪）
            n_views: 视图数量 U
            n_depths: 深度切片数 Z
            feat_ch: 特征通道数 C
        """
        super().__init__()
        self.n_views = n_views
        self.n_depths = n_depths

        # 注册 Gaussian 软核为 buffer（不参与梯度更新）
        self.register_buffer('gauss_kernels', gauss_warp_psfs)  # (U, Z, H_k, W_k)

        # 可学习的逐深度权重调整参数预测头
        # 输入: 特征图 (B*U, C, H, W) → 输出: 逐像素的深度权重调整 (B*U, Z, H, W)
        self.depth_weight_head = nn.Sequential(
            nn.Conv2d(feat_ch, feat_ch, 3, padding=1),
            nn.LeakyReLU(0.01, True),
            nn.Conv2d(feat_ch, n_depths, 1),
        )

        # 初始化预测头使初始输出接近 0（残差设计）
        # 这样 sigmoid(0) = 0.5，乘以 2 后 = 1.0，等价于不调整
        nn.init.zeros_(self.depth_weight_head[-1].weight)
        nn.init.zeros_(self.depth_weight_head[-1].bias)

    def _gauss_warp(self, feats):
        """用 Gaussian 软核对特征做 FFT 卷积对齐（替代原始 warp_feats）。

        Args:
            feats: (U, C, H, W) 逐视图特征
        Returns:
            warped: (U, C, Z, H, W) 对齐后的 3D 特征体
        """
        u, ch, ra, ca = feats.shape
        _, z, rb, cb = self.gauss_kernels.shape

        r = ra + rb - 1
        p1 = (r - ra) / 2

        a1 = torch.zeros(u, ch, 1, r, r, device=feats.device)
        b1 = torch.zeros(u, 1, z, r, r, device=feats.device)

        a1[:, :, :, 0:ra, 0:ca] = feats.unsqueeze(2)
        b1[:, :, :, 0:rb, 0:cb] = self.gauss_kernels.unsqueeze(1)

        projections = ifft2(fft2(a1) * fft2(b1))
        projections = torch.real(projections[:, :, :, int(p1):int(r - p1), int(p1):int(r - p1)])
        return projections

    def forward(self, feats):
        """
        Args:
            feats: (U, C, H, W) 逐视图提取的特征

        Returns:
            aligned: (U, C, Z, H, W) 对齐后的 3D 特征体（与原始 warp_feats 输出形状一致）
        """
        U, C, H, W = feats.shape

        # Step 1: 用 Gaussian 软核做基础对齐
        warped = self._gauss_warp(feats)   # (U, C, Z, H, W)

        # Step 2: 预测逐深度的权重调整
        depth_logits = self.depth_weight_head(feats)   # (U, Z, H, W)
        # 残差设计：sigmoid(0)=0.5, ×2=1.0 → 初始不改变权重
        depth_weights = 2.0 * torch.sigmoid(depth_logits)   # (U, Z, H, W) 范围 (0, 2)
        depth_weights = depth_weights.unsqueeze(1)   # (U, 1, Z, H, W) 广播到 C 维

        # Step 3: 加权调整
        aligned = warped * depth_weights

        return aligned


class Unet(nn.Module):
    def __init__(self,n_slices,input_channel):
        super().__init__()
        channels_interp = 128
        
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channel,out_channels=8*channels_interp,kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(0.01), 
            nn.Conv2d(in_channels=8*channels_interp,out_channels=4*channels_interp,kernel_size=3,stride=1,padding=1), 
            nn.LeakyReLU(0.01),
            nn.Conv2d(in_channels=4*channels_interp,out_channels=2*channels_interp,kernel_size=3,stride=1,padding=1), 
            nn.LeakyReLU(0.01),
            nn.Conv2d(in_channels=2*channels_interp,out_channels=channels_interp,kernel_size=3,stride=2,padding=1), 
            nn.LeakyReLU(0.01))

        self.encoder_1 = Encoder(in_channels=channels_interp,out_channels=128)
        self.encoder_2 = Encoder(in_channels=128,out_channels=256)
        self.encoder_3 = Encoder(in_channels=256,out_channels=384)
        self.encoder_4 = Encoder(in_channels=384,out_channels=512)

        self.upscale_2 = Upscale(in_channels=512,out_channels=512,mode='upconv',scale=2)

        self.decoder_1 = Decoder(in_channels=896,out_channels=384,scale=2)
        self.decoder_2 = Decoder(in_channels=640,out_channels=256,scale=2)
        self.decoder_3 = Decoder(in_channels=384,out_channels=128,scale=2)
        self.decoder_4 = Decoder(in_channels=256,out_channels=128,scale=2)

        self.conv_final = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(0.01), 
            nn.Conv2d(in_channels=256,out_channels=128,kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(0.01), 
            nn.Conv2d(in_channels=128,out_channels=n_slices,kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(0.01))

    def forward(self,x):
        #upscale
        f = self.init_conv(x)
        
        #encoder
        encoder_layers = []
        encoder_layers.append(f)
        f = self.encoder_1(f)
        encoder_layers.append(f)
        f = self.encoder_2(f)
        encoder_layers.append(f)
        f = self.encoder_3(f)
        encoder_layers.append(f)
        f = self.encoder_4(f)

        #decoder
        f = F.leaky_relu(f,0.05)
        f = self.upscale_2(f)
        f = torch.cat((encoder_layers[3],f),dim=1)
        f = self.decoder_1(f)
        f = torch.cat((encoder_layers[2],f),dim=1)
        f = self.decoder_2(f)
        f = torch.cat((encoder_layers[1],f),dim=1)
        f = self.decoder_3(f)
        f = torch.cat((encoder_layers[0],f),dim=1)
        f = self.decoder_4(f)
        f = self.conv_final(f)

        return f
    
class V2V3D(nn.Module):
    def __init__(self,warp_psfs,n_slice,select_v,remain_v,use_views=13,feat_ch=4,
                 gauss_warp_psfs=None):
        super().__init__()
        self.use_v = use_views
        self.feat_ch = feat_ch
        self.select_v = select_v
        self.remain_v = remain_v

        self.use_gauss_align = gauss_warp_psfs is not None

        if self.use_gauss_align:
            self.gauss_align = GaussianFeatureAlignment(
                gauss_warp_psfs, use_views, n_slice, feat_ch)
        else:
            self.warp_psfs = warp_psfs

        self.feat_extract = Feature(in_channels=1,out_channels=feat_ch)
        self.unet1 = Unet(n_slices=n_slice,input_channel=feat_ch*select_v.shape[0]*n_slice)
        self.unet2 = Unet(n_slices=n_slice,input_channel=feat_ch*remain_v.shape[0]*n_slice)
        self.weight_init(mean=0.0, std=0.02)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self,x):

        V,H,W = x.shape
        x = torch.unsqueeze(x,1) #v,c,h,w
        feats = self.feat_extract(x)

        if self.use_gauss_align:
            feats = self.gauss_align(feats)
        else:
            feats = warp_feats(self.warp_psfs,feats)

        volume1 = self.unet1(feats[self.select_v,...].reshape(1,len(self.select_v)*feats.shape[1]*feats.shape[2],H,W))
        volume2 = self.unet2(feats[self.remain_v,...].reshape(1,len(self.remain_v)*feats.shape[1]*feats.shape[2],H,W))
        volume = (volume1+volume2)/2

        return volume1,volume2,volume