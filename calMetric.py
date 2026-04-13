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

folder1_path = 'Dataset/Samples_test/'  # Ground Truth 文件夹
folder2_path = 'Results/01_16_20_50_V2V3D_F4_LFIs_test/'   # Recon 文件夹 (文件名带有 _recon 后缀)
#folder2_path = 'Results/01_21_11_34_V2V3D_F4_LFIs_test/'
ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
psnr = PeakSignalNoiseRatio(data_range=1.0)

psnr_list = []
ssim_list = []

file_list = os.listdir(folder1_path)
file_list.sort()

print(f"{'GT Filename':<25} | {'Recon Filename':<25} | {'PSNR':<8} | {'SSIM':<8}")
print("-" * 75)

for filename in file_list:
    # 过滤非 tif 文件
    if not (filename.endswith('.tif') or filename.endswith('.tiff')):
        continue

    # --- 核心修改部分 START ---
    # 1. 分离文件名和扩展名 (例如: 'Vessels_2.tif' -> 'Vessels_2' 和 '.tif')
    name_part, ext_part = os.path.splitext(filename)
    
    # 2. 构建带后缀的文件名 (例如: 'Vessels_2_recon.tif')
    recon_filename = f"{name_part}_recon{ext_part}"
    
    # 3. 拼接完整路径
    img1_fullpath = os.path.join(folder1_path, filename)       # 原文件名
    img2_fullpath = os.path.join(folder2_path, recon_filename) # 加了后缀的文件名
    # --- 核心修改部分 END ---

    # 检查对应的 _recon 文件是否存在
    if not os.path.exists(img2_fullpath):
        print(f"File {recon_filename} not found in reconstruction folder (matching {filename}).")
        continue

    try:
        img1 = torch.from_numpy(tf.imread(img1_fullpath).astype(np.float32)).squeeze()
        img2 = torch.from_numpy(tf.imread(img2_fullpath).astype(np.float32)).squeeze()

        img1 = img1[:, 10:-10, 10:-10]
        img2 = img2[:, 10:-10, 10:-10]

        if img1.shape != img2.shape:
            print(f"Shape mismatch: {filename} vs {recon_filename}")
            continue

        img1 = relu(img1 - calBacknoise(img1))
        img2 = relu(img2 - calBacknoise(img2))

        if img1.ndim == 2: img1.unsqueeze_(0)
        if img2.ndim == 2: img2.unsqueeze_(0)

        # Calculate Metrics
        psnr_value = psnr(img2.unsqueeze(0), img1.unsqueeze(0))
        ssim_value = ssim(img2.unsqueeze(0), img1.unsqueeze(0))

        # 打印时显示两个文件名，方便确认匹配是否正确
        print(f"{filename:<25} | {recon_filename:<25} | {psnr_value.item():.4f}   | {ssim_value.item():.4f}")

        psnr_list.append(psnr_value.item())
        ssim_list.append(ssim_value.item())

    except Exception as e:
        print(f"Error processing {filename}: {e}")

print("-" * 75)
if len(psnr_list) > 0:
    avg_psnr = sum(psnr_list) / len(psnr_list)
    avg_ssim = sum(ssim_list) / len(ssim_list)
    print(f"Average PSNR: {avg_psnr:.4f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
else:
    print("No valid image pairs found.")