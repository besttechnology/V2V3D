import tifffile as tf
import numpy as np
import torch
from torch.nn.functional import relu
from torchmetrics.image import StructuralSimilarityIndexMeasure,PeakSignalNoiseRatio
import os

def calBacknoise(img,bins=100):
    hist_max = torch.mean(img).item()
    hist_min = torch.min(img).item()
    hist = torch.histc(img,bins,hist_min,hist_max)
    backnoise_index = torch.max(hist,0)[1]+1
    backnoise = (backnoise_index/bins)*(hist_max-hist_min)+hist_min

    return backnoise

img1_path = 'Dataset/Samples_test/Vessels_2.tif' #sample path
img2_path = 'Dataset/V2V3D_recon/Vessels_2.tif' #recon path

ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
psnr = PeakSignalNoiseRatio(data_range=1.0)

img1 = torch.from_numpy(tf.imread(img1_path).astype(np.float32)).squeeze()
img2 = torch.from_numpy(tf.imread(img2_path).astype(np.float32)).squeeze()

img1 = img1[:,10:-10,10:-10] #Crop edge
img2 = img2[:,10:-10,10:-10]

img1_mean = torch.mean(img1)
img2_mean = torch.mean(img2)
print('gt mean:',img1_mean.item(),'recon mean:',img2_mean.item())

img1 = relu(img1-calBacknoise(img1)) #Subtract backnoise
img2 = relu(img2-calBacknoise(img2))

if img1.ndim == 2: img1.unsqueeze_(0)
if img2.ndim == 2: img2.unsqueeze_(0)

# Calculate PSNR
psnr_value = psnr(img2.unsqueeze(0),img1.unsqueeze(0))
print(f"PSNR: {psnr_value} dB")

# Calculate SSIM
ssim_value = ssim(img2.unsqueeze(0),img1.unsqueeze(0))
print(f"SSIM: {ssim_value}")
