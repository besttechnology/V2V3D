"""参数化评估：对给定 recon 文件夹，按 calMetric_v2v3d 的逻辑算 PSNR/SSIM。
用法: python eval_folder.py <recon_dir> [gt_dir]
"""
import sys, os
import tifffile as tf
import numpy as np
import torch
from torch.nn.functional import relu
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

def calBacknoise(img, bins=100):
    hist_max = torch.mean(img).item()
    hist_min = torch.min(img).item()
    hist = torch.histc(img, bins, hist_min, hist_max)
    backnoise_index = torch.max(hist, 0)[1] + 1
    backnoise = (backnoise_index / bins) * (hist_max - hist_min) + hist_min
    return backnoise

recon_dir = sys.argv[1]
gt_dir = sys.argv[2] if len(sys.argv) > 2 else 'Dataset/Samples_test'

ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
psnr = PeakSignalNoiseRatio(data_range=1.0)

file_list = sorted(f for f in os.listdir(gt_dir) if f.endswith(('.tif', '.tiff')))
print(f"{'Filename':<22} | {'PSNR':<9} | {'SSIM':<9}")
print("-" * 48)
psnr_list, ssim_list = [], []
for filename in file_list:
    p1 = os.path.join(gt_dir, filename)
    p2 = os.path.join(recon_dir, filename)
    if not os.path.exists(p2):
        print(f"{filename:<22} | MISSING")
        continue
    img1 = torch.from_numpy(tf.imread(p1).astype(np.float32)).squeeze()
    img2 = torch.from_numpy(tf.imread(p2).astype(np.float32)).squeeze()
    img1 = img1[:, 10:-10, 10:-10]
    img2 = img2[:, 10:-10, 10:-10]
    if img1.shape != img2.shape:
        print(f"{filename:<22} | shape mismatch {img1.shape} vs {img2.shape}")
        continue
    img1 = relu(img1 - calBacknoise(img1))
    img2 = relu(img2 - calBacknoise(img2))
    if img1.ndim == 2: img1.unsqueeze_(0)
    if img2.ndim == 2: img2.unsqueeze_(0)
    pv = psnr(img2.unsqueeze(0), img1.unsqueeze(0)).item()
    sv = ssim(img2.unsqueeze(0), img1.unsqueeze(0)).item()
    print(f"{filename:<22} | {pv:8.4f} | {sv:8.4f}")
    psnr_list.append(pv); ssim_list.append(sv)
print("-" * 48)
if psnr_list:
    print(f"Average PSNR: {sum(psnr_list)/len(psnr_list):.4f} dB")
    print(f"Average SSIM: {sum(ssim_list)/len(ssim_list):.4f}")
    print(f"N = {len(psnr_list)}")
