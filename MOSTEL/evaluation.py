import math
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from scipy import signal, ndimage
from tqdm import tqdm
from pytorch_fid import fid_score
from eval_utils import devdata, fspecial_gauss


parser = argparse.ArgumentParser()
parser.add_argument('--target_path', type=str, default='', help='results')
parser.add_argument('--gt_path', type=str, default='', help='labels')
parser.add_argument('--no_fid', action='store_true', default=False)
args = parser.parse_args()
img_path = args.target_path
gt_path = args.gt_path

sum_psnr = 0
sum_ssim = 0
sum_mse = 0
count = 0
sum_time = 0.0
l1_loss = 0


def ssim(img1, img2, cs_map=False):
    """Return the Structural Similarity Map corresponding to input images img1
    and img2 (images are assumed to be uint8)

    This function attempts to mimic precisely the functionality of ssim.m a
    MATLAB provided by the author's of SSIM
    https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m
    """
    img1 = img1.astype(float)
    img2 = img2.astype(float)

    size = min(img1.shape[0], 11)
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 255  # bitdepth of image
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    mu1 = signal.fftconvolve(img1, window, mode='valid')
    mu2 = signal.fftconvolve(img2, window, mode='valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = signal.fftconvolve(img1 * img1, window, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(img2 * img2, window, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(img1 * img2, window, mode='valid') - mu1_mu2
    if cs_map:
        return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)),
                (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
    else:
        return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))


def msssim(img1, img2):
    """This function implements Multi-Scale Structural Similarity (MSSSIM) Image
    Quality Assessment according to Z. Wang's "Multi-scale structural similarity
    for image quality assessment" Invited Paper, IEEE Asilomar Conference on
    Signals, Systems and Computers, Nov. 2003

    Author's MATLAB implementation:-
    http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
    """
    level = 5
    weight = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    downsample_filter = np.ones((2, 2)) / 4.0
    mssim = np.array([])
    mcs = np.array([])
    for l in range(level):
        ssim_map, cs_map = ssim(img1, img2, cs_map=True)
        mssim = np.append(mssim, ssim_map.mean())
        mcs = np.append(mcs, cs_map.mean())
        filtered_im1 = ndimage.filters.convolve(img1, downsample_filter,
                                                mode='reflect')
        filtered_im2 = ndimage.filters.convolve(img2, downsample_filter,
                                                mode='reflect')
        im1 = filtered_im1[:: 2, :: 2]
        im2 = filtered_im2[:: 2, :: 2]

    # Note: Remove the negative and add it later to avoid NaN in exponential.
    sign_mcs = np.sign(mcs[0: level - 1])
    sign_mssim = np.sign(mssim[level - 1])
    mcs_power = np.power(np.abs(mcs[0: level - 1]), weight[0: level - 1])
    mssim_power = np.power(np.abs(mssim[level - 1]), weight[level - 1])
    return np.prod(sign_mcs * mcs_power) * sign_mssim * mssim_power


imgData = devdata(dataRoot=img_path, gtRoot=gt_path)
data_loader = DataLoader(
    imgData,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    drop_last=False)

for idx, (img, lbl, path) in tqdm(enumerate(data_loader), total=len(data_loader)):
    mse = ((lbl - img)**2).mean()
    sum_mse += mse
    if mse == 0:
        continue
    count += 1
    
    psnr = 10 * math.log10(1 / mse)
    sum_psnr += psnr
    
    R = lbl[0, 0, :, :]
    G = lbl[0, 1, :, :]
    B = lbl[0, 2, :, :]
    YGT = .299 * R + .587 * G + .114 * B
    R = img[0, 0, :, :]
    G = img[0, 1, :, :]
    B = img[0, 2, :, :]
    YBC = .299 * R + .587 * G + .114 * B
    mssim = msssim(np.array(YGT * 255), np.array(YBC * 255))
    sum_ssim += mssim

print('PSNR:', sum_psnr / count)
print('SSIM:', sum_ssim / count)
print('MSE:', sum_mse.item() / count)

if not args.no_fid:
    batch_size = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dims = 2048
    fid_value = fid_score.calculate_fid_given_paths([str(gt_path), str(img_path)], batch_size, device, dims)
    print('FID:', fid_value)
