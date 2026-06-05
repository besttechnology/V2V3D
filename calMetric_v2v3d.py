import tifffile as tf
import numpy as np
import torch
from torch.nn.functional import relu
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import os

def calBacknoise(img, bins=100):
    hist_max = torch.mean(img).item()
    hist_min = torch.min(img).item()
    hist = torch.histc(img, bins, hist_min, hist_max)
    backnoise_index = torch.max(hist, 0)[1] + 1
    backnoise = (backnoise_index / bins) * (hist_max - hist_min) + hist_min
    return backnoise

# 1. 修改为文件夹路径 (请确保路径末尾没有文件名)
folder1_path = 'Dataset/Samples_test/'  # Ground Truth folder
#folder2_path = 'Dataset/V2V3D_recon/'   # Recon folder
folder2_path = 'Results/06_04_20_47_V2V3D_F4_LFIs_test'   # Recon folder

ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
psnr = PeakSignalNoiseRatio(data_range=1.0)

# 用于存储所有文件的结果以计算平均值
psnr_list = []
ssim_list = []

# 获取文件夹下的文件列表
file_list = os.listdir(folder1_path)
file_list.sort()

print(f"{'Filename':<30} | {'PSNR':<10} | {'SSIM':<10}")
print("-" * 55)

for filename in file_list:
    # 过滤非 tif 文件，防止报错
    if not (filename.endswith('.tif') or filename.endswith('.tiff')):
        continue

    img1_fullpath = os.path.join(folder1_path, filename)
    img2_fullpath = os.path.join(folder2_path, filename)

    # 检查第二个文件夹中是否存在对应文件
    if not os.path.exists(img2_fullpath):
        print(f"File {filename} not found in reconstruction folder.")
        continue

    try:
        # 读取图像
        img1 = torch.from_numpy(tf.imread(img1_fullpath).astype(np.float32)).squeeze()
        img2 = torch.from_numpy(tf.imread(img2_fullpath).astype(np.float32)).squeeze()

        # 保持原有的裁剪逻辑 (Crop edge)
        img1 = img1[:, 10:-10, 10:-10]
        img2 = img2[:, 10:-10, 10:-10]

        # 简单的形状检查，防止计算报错
        if img1.shape != img2.shape:
            print(f"Shape mismatch for {filename}: {img1.shape} vs {img2.shape}")
            continue

        # 保持原有的去噪逻辑 (Subtract backnoise)
        img1 = relu(img1 - calBacknoise(img1))
        img2 = relu(img2 - calBacknoise(img2))

        if img1.ndim == 2: img1.unsqueeze_(0)
        if img2.ndim == 2: img2.unsqueeze_(0)

        # Calculate PSNR
        # 注意：metric计算通常需要增加 batch 维度 (unsqueeze(0))
        psnr_value = psnr(img2.unsqueeze(0), img1.unsqueeze(0))
        
        # Calculate SSIM
        ssim_value = ssim(img2.unsqueeze(0), img1.unsqueeze(0))

        # 打印单张图片结果
        print(f"{filename:<30} | {psnr_value.item():.4f}     | {ssim_value.item():.4f}")

        # 存入列表
        psnr_list.append(psnr_value.item())
        ssim_list.append(ssim_value.item())

    except Exception as e:
        print(f"Error processing {filename}: {e}")

# 打印平均结果
print("-" * 55)
if len(psnr_list) > 0:
    avg_psnr = sum(psnr_list) / len(psnr_list)
    avg_ssim = sum(ssim_list) / len(ssim_list)
    print(f"Average PSNR: {avg_psnr:.4f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
else:
    print("No valid image pairs found.")