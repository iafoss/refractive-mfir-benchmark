
from argparse import ArgumentParser
import glob, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from tqdm.auto import tqdm
from collections import OrderedDict

from data import MFIR_Dataset, gen_video, MFIR_Dataset_gt
from metrics import LPIPS_metric, ZS_IQA
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from img_gen import gen_img_ref, gen_img_mean, gen_DATUM, gen_Grid_registration, gen_FILE, gen_VIDEO, gen_UNSUPERVISED_NDIR

amplitudes = {
    'ocean':{'low':0.329, 'mid':0.574, 'high':0.983, 'extreme':1.642},
    'shallow':{'low':0.138, 'mid':0.244, 'high':0.433, 'extreme':0.695},
    'sine':{'low':0.204, 'mid':0.358, 'high':0.617, 'extreme':1.027},
    'ripple':{'low':0.315, 'mid':0.548, 'high':0.943, 'extreme':1.6},
}
paths = {
    'ocean': 'Ocean_waves', 'shallow': 'Shallow_waves',
    'sine': 'Sine_waves', 'ripple': 'Ripples',
}

def _run(args):
    metric_bs=32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    amplitude = amplitudes[args.wave_type][args.amplitude]

    path_w = os.path.join(args.dataset_root,'wave_profiles',paths[args.wave_type])
    path_b = os.path.join(args.dataset_root,'backgrounds')

    ds_gt = MFIR_Dataset_gt(path_w=path_w, path_b=path_b)
    ds = MFIR_Dataset(path_w=path_w, path_b=path_b, depth=amplitude, scale=amplitude, fast=True, L=args.L)

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    lpips_alex_metric = LPIPS_metric(device=device)
    lpips_vgg_metric = LPIPS_metric(device=device, net='vgg')

    zs_iqa_dino = ZS_IQA('dinov1', device)
    zs_iqa_clip = ZS_IQA('clip_vitb32', device)
    if args.method == 'ref': gen_img = gen_img_ref
    elif args.method == 'mean': gen_img = gen_img_mean
    elif args.method == 'grid_registration': gen_img = gen_Grid_registration()
    elif args.method == 'datum': gen_img = gen_DATUM()
    elif args.method == 'file': gen_img = gen_FILE(path=args.path_eval)
    elif args.method == 'video': gen_img = gen_VIDEO(path=args.path_eval, L=args.L)
    elif args.method == 'grid_deformation': gen_img = gen_UNSUPERVISED_NDIR(path=args.path_eval, L=args.L)
    else: raise NotImplementedError(f'Invalid method {args.method}')

    scores = []
    for k in tqdm(range(len(ds))):
        x0 = ds[k]
        if 'ref_video' not in x0:
            x0['ref_video'] = gen_video(x0['video'].unsqueeze(0).to(device), x0['norms'].unsqueeze(0).to(device), 
                                 scale=amplitude, depth=amplitude)[0]
        pred = gen_img(x0,k).float().to(device)/255 #L C H W [0..255]

        gt = cv2.imread(ds_gt[k])[:,:,::-1]
        gt = torch.from_numpy(gt.astype(np.float32)/255).permute(2,0,1)
        gt = gt.unsqueeze(0).expand(pred.shape[0],-1,-1,-1).to(device)

        with torch.no_grad():
            psnr = psnr_metric(pred, gt).cpu().item()
            ssim = ssim_metric(pred, gt).cpu().item()
            lpips_alex,lpips_vgg = [],[]
            B = pred.shape[0]
            
            for s in range(0, B, metric_bs):
                lpips_alex.append(lpips_alex_metric(pred[s:min(s+metric_bs,B)], gt[s:min(s+metric_bs,B)]).cpu())
                lpips_vgg.append(lpips_vgg_metric(pred[s:min(s+metric_bs,B)], gt[s:min(s+metric_bs,B)]).cpu())
            lpips_alex = torch.cat(lpips_alex).mean().item()
            lpips_vgg = torch.cat(lpips_vgg).mean().item()

            with torch.no_grad():
                metric_dino = 0,0
                metric_clip = 0,0
                B = pred.shape[0]
                for s in range(0, B, metric_bs):
                    metric_dino += zs_iqa_dino.get_score(pred[s:min(s+metric_bs,B)], gt[s:min(s+metric_bs,B)])
                metric_dino /= B
                for s in range(0, B, 1):
                    metric_clip += zs_iqa_clip.get_score(pred[s:min(s+1,B)], gt[s:min(s+1,B)])
                metric_clip /= B
            
        scores.append({
            'idx': k,
            'psnr': psnr, 'ssim': ssim, 'lpips_alex': lpips_alex, 'lpips_vgg': lpips_vgg,
            'dino': metric_dino, 'clip': metric_clip, 
        })
        
    psnr = np.array([el['psnr'] for el in scores]).mean()
    ssim = np.array([el['ssim'] for el in scores]).mean()
    lpips_alex = np.array([el['lpips_alex'] for el in scores]).mean()
    lpips_vgg = np.array([el['lpips_vgg'] for el in scores]).mean()
    metric_dino = np.array([el['dino'] for el in scores]).mean()
    metric_clip = np.array([el['clip'] for el in scores]).mean()
    print(f'PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}, LPIPS vgg {lpips_vgg:.4f}, LPIPS alex {lpips_alex:.4f}, DINO {metric_dino:.4f}, CLIP {metric_clip:.4f}')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset_root', type=str, default='./dataset_refractive_mfir_benchmark') # path to wave profiles and backgrounds
    parser.add_argument('--L', type=int, default=1)                     # number of frames
    parser.add_argument('--wave_type', type=str, default='ocean')       # wave type: ocean | shallow | sine | ripple
    parser.add_argument('--amplitude', type=str, default='low')         # wave amplitude: low | mid | high | extreme
    parser.add_argument('--method', type=str, default='ref')            # method: ref | mean | grid_registration | datum | grid_deformation
    parser.add_argument('--path_eval', type=str, default='')            # path for eval on image or video (method file | video)
    args = parser.parse_args()
    _run(args)
