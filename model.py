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


class GaussianUnprojection(nn.Module):
    """完整的 Gaussian Unprojection 特征对齐模块。

    将每个视图的 2D 特征像素视为 Gaussian Splat，通过可学习的 3D 参数
    反投影到统一的 3D 特征空间。与 3DGS 的核心对应关系：

    - 空间卷积核 = Gaussian Splat 的 2D 投影（位置 μ、尺度 σ 可学习）
    - 深度权重 = Gaussian Splat 沿 z 轴的分布（中心、宽度可学习）
    - 置信度 = Gaussian Splat 的不透明度 α

    三组参数均采用残差设计，从 PSF 物理先验初始化：
    - 空间核: centroid, sigma 初始化自 PSF 二阶矩拟合
    - 深度权重: 初始近似平坦（不偏好任何深度）
    - 置信度: 初始 = 1.0（不衰减）

    初始行为完全等价于固定 Gaussian 软核对齐，训练中逐步学习优化。
    """

    def __init__(self, psf_priors, n_views, n_depths, feat_ch):
        """
        Args:
            psf_priors: dict, 包含 centroid (U,Z,2), sigma (U,Z,2),
                        crop_min (int), kernel_size (H_k, W_k)
            n_views: 视图数量 U
            n_depths: 深度切片数 Z
            feat_ch: 特征通道数 C
        """
        super().__init__()
        self.n_views = n_views
        self.n_depths = n_depths
        self.kernel_h, self.kernel_w = psf_priors['kernel_size']

        # ========== 空间对齐：可学习 Gaussian 卷积核 ==========
        # PSF 先验（转换到裁剪坐标系）
        crop_min = psf_priors['crop_min']
        centroid = psf_priors['centroid'].clone()
        centroid[:, :, 0] -= crop_min
        centroid[:, :, 1] -= crop_min

        self.register_buffer('prior_cx', centroid[:, :, 0].contiguous())   # (U, Z)
        self.register_buffer('prior_cy', centroid[:, :, 1].contiguous())   # (U, Z)
        self.register_buffer('prior_sx', psf_priors['sigma'][:, :, 0].contiguous())  # (U, Z)
        self.register_buffer('prior_sy', psf_priors['sigma'][:, :, 1].contiguous())  # (U, Z)

        # 可学习残差：空间核参数
        self.delta_cx = nn.Parameter(torch.zeros(n_views, n_depths))
        self.delta_cy = nn.Parameter(torch.zeros(n_views, n_depths))
        self.delta_log_sx = nn.Parameter(torch.zeros(n_views, n_depths))   # log 缩放，exp(0)=1
        self.delta_log_sy = nn.Parameter(torch.zeros(n_views, n_depths))

        # ========== 深度权重：逐像素 Gaussian-in-z ==========
        # 预测头（视图间共享权重）
        self.param_head = nn.Sequential(
            nn.Conv2d(feat_ch, feat_ch, 3, padding=1),
            nn.LeakyReLU(0.01, True),
            nn.Conv2d(feat_ch, 3, 1),   # [delta_d, delta_log_w, logit_alpha]
        )
        # 残差初始化：输出全 0
        nn.init.zeros_(self.param_head[-1].weight)
        nn.init.zeros_(self.param_head[-1].bias)

        # z 轴索引（用于计算 Gaussian-in-z）
        self.register_buffer('z_indices', torch.arange(n_depths, dtype=torch.float32))

    def _build_kernels(self):
        """从可学习参数动态生成 Gaussian 软卷积核。

        Returns:
            kernels: (U, Z, H_k, W_k) 归一化的 Gaussian 核
        """
        # 残差叠加：prior + learnable delta
        cx = self.prior_cx + self.delta_cx     # (U, Z)
        cy = self.prior_cy + self.delta_cy
        sx = (self.prior_sx * torch.exp(self.delta_log_sx)).clamp(min=0.3)   # (U, Z)
        sy = (self.prior_sy * torch.exp(self.delta_log_sy)).clamp(min=0.3)

        # 坐标网格
        x_grid = torch.arange(self.kernel_w, device=cx.device, dtype=torch.float32)   # (W_k,)
        y_grid = torch.arange(self.kernel_h, device=cy.device, dtype=torch.float32)   # (H_k,)

        # 可分离 Gaussian：x 方向 (U,Z,W_k) 和 y 方向 (U,Z,H_k)
        gx = torch.exp(-0.5 * ((x_grid[None, None, :] - cx[:, :, None]) / sx[:, :, None]) ** 2)
        gy = torch.exp(-0.5 * ((y_grid[None, None, :] - cy[:, :, None]) / sy[:, :, None]) ** 2)

        # 外积 → 2D Gaussian 核 (U, Z, H_k, W_k)
        kernels = gy.unsqueeze(-1) * gx.unsqueeze(-2)
        kernels = kernels / (kernels.sum(dim=(-2, -1), keepdim=True) + 1e-12)
        return kernels

    def _fft_warp(self, feats, kernels):
        """用动态生成的 Gaussian 核对特征做 FFT 卷积对齐。

        Args:
            feats: (U, C, H, W)
            kernels: (U, Z, H_k, W_k)
        Returns:
            warped: (U, C, Z, H, W)
        """
        u, ch, ra, ca = feats.shape
        _, z, rb, cb = kernels.shape

        r = ra + rb - 1
        p1 = (r - ra) / 2

        a1 = torch.zeros(u, ch, 1, r, r, device=feats.device)
        b1 = torch.zeros(u, 1, z, r, r, device=feats.device)

        a1[:, :, :, 0:ra, 0:ca] = feats.unsqueeze(2)
        b1[:, :, :, 0:rb, 0:cb] = kernels.unsqueeze(1)

        projections = ifft2(fft2(a1) * fft2(b1))
        projections = torch.real(projections[:, :, :, int(p1):int(r - p1), int(p1):int(r - p1)])
        return projections

    def forward(self, feats):
        """
        Args:
            feats: (U, C, H, W) 逐视图提取的特征

        Returns:
            aligned: (U, C, Z, H, W) 对齐后的 3D 特征体
        """
        U, C, H, W = feats.shape
        Z = self.n_depths

        # Step 1: 动态生成可学习 Gaussian 空间卷积核
        kernels = self._build_kernels()   # (U, Z, H_k, W_k)

        # Step 2: FFT 卷积实现空间对齐
        warped = self._fft_warp(feats, kernels)   # (U, C, Z, H, W)

        # Step 3: 逐像素深度权重预测（Gaussian-in-z）
        params = self.param_head(feats)            # (U, 3, H, W)
        delta_d = params[:, 0:1]                   # (U, 1, H, W) 深度中心偏移
        delta_log_w = params[:, 1:2]               # (U, 1, H, W) 深度宽度缩放
        logit_alpha = params[:, 2:3]               # (U, 1, H, W) 置信度

        # 深度 Gaussian 参数（残差设计）
        depth_center = (Z / 2.0) + delta_d                       # 初始 = Z/2
        depth_width = (Z * 10.0) * torch.exp(delta_log_w)        # 初始 = Z*10（近似平坦）
        confidence = torch.sigmoid(logit_alpha)                   # 初始 = 0.5

        # 扩展维度用于广播: (U, 1, 1, H, W) vs z_idx (1, 1, Z, 1, 1)
        depth_center = depth_center.unsqueeze(2)                  # (U, 1, 1, H, W)
        depth_width = depth_width.unsqueeze(2)                    # (U, 1, 1, H, W)
        confidence = confidence.unsqueeze(2)                      # (U, 1, 1, H, W)
        z_idx = self.z_indices.view(1, 1, Z, 1, 1)               # (1, 1, Z, 1, 1)

        # Gaussian-in-z 权重，乘以 2×confidence 使初始值 ≈ 1.0
        depth_weights = torch.exp(-0.5 * ((z_idx - depth_center) / (depth_width + 1e-6)) ** 2)
        depth_weights = 2.0 * confidence * depth_weights          # (U, 1, Z, H, W)

        # Step 4: 加权调整
        aligned = warped * depth_weights   # (U, C, Z, H, W)

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
                 psf_priors=None):
        super().__init__()
        self.use_v = use_views
        self.feat_ch = feat_ch
        self.select_v = select_v
        self.remain_v = remain_v

        self.use_gauss_align = psf_priors is not None

        if self.use_gauss_align:
            self.gauss_align = GaussianUnprojection(
                psf_priors, use_views, n_slice, feat_ch)
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