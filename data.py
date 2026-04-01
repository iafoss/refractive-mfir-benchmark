import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from transformers import AutoTokenizer
import pickle, os
import pandas as pd
import numpy as np
from torchvision import transforms
import random
import cv2
import albumentations as A
import warnings
warnings.filterwarnings("ignore")
import torch
from PIL import Image 

def load_OpenCV(f): 
    arr = cv2.imread(f)[:,:,::-1] #,cv2.IMREAD_UNCHANGED)[:,:,::-1] 
    return arr
    
def load_PIL(f): 
    im = Image.open(f).convert('RGB')
    return np.asarray(im) 

def get_refraction(norm_r,norm_s,n=1.33):
    return n*torch.cross(norm_s, torch.cross(-norm_s, norm_r, dim=-1), dim=-1) - \
     norm_s*torch.sqrt(1 - torch.pow(n*torch.linalg.norm(torch.cross(norm_s, norm_r, dim=-1), dim=-1),2)).unsqueeze(-1)


def get_aug_all(L=16, crop_sz=None):
    ags = [
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.RandomRotate90(), 
    ]
    if crop_sz is not None:
        ags.append(RandomCrop(crop_sz,crop_sz))
    return A.Compose(ags, p=1, additional_targets={f'image{i}':'image' for i in range(L)})
    
def get_aug_all_crop(L=16, crop_sz=256, random_crop=False):
    ags = [A.RandomCrop(crop_sz,crop_sz)] if random_crop else A.CenterCrop(crop_sz,crop_sz)
    if crop_sz is not None:
        return A.Compose(ags, p=1, additional_targets={f'image{i}':'image' for i in range(L)})   
    else: return None

def aug_train(sz=(512,512)):
    return A.Compose([
            A.PadIfNeeded(min_height=sz[0], min_width=sz[1], border_mode=cv2.BORDER_CONSTANT, value=0, p=1),
            A.RandomCrop(sz[0],sz[1],p=1),
            A.HorizontalFlip()
        ],p=1)
def aug_val(sz=(512,512)):
    return A.Compose([
            A.PadIfNeeded(min_height=sz[0], min_width=sz[1], border_mode=cv2.BORDER_CONSTANT, value=0, p=1),
            A.CenterCrop(sz[0],sz[1],p=1),
        ],p=1)

@torch.no_grad
@torch.autocast("cpu", enabled=False)
@torch.autocast("cuda", enabled=False)
def get_refraction(norm_r,norm_s,n=1.33):
    return n*torch.cross(norm_s, torch.cross(-norm_s, norm_r, dim=-1), dim=-1) - \
      norm_s*torch.nan_to_num(torch.sqrt(1 - torch.pow(n*torch.linalg.norm(torch.cross(norm_s, norm_r, dim=-1), dim=-1),2))).unsqueeze(-1)

@torch.no_grad
@torch.autocast("cpu", enabled=False)
@torch.autocast("cuda", enabled=False)
def gen_video(x, norm_s, scale=[0.75,1.25], depth=[0.75,1.25], n=1.33, mode='val', x2_sample=False): # B L C H W
    if not isinstance(scale,list) and not isinstance(scale,set) and not isinstance(scale,tuple): scale = (scale,scale)
    if not isinstance(depth,list) and not isinstance(depth,set) and not isinstance(depth,tuple): depth = (depth,depth)
    B,_,_,H,W = x.shape
    _,L,_,h,w = norm_s.shape
    device = x.device
    
    idx_i, idx_j = torch.arange(H, device=device)[:,None], torch.arange(W, device=device)[None,:]
    idx_i, idx_j = (idx_i/(H-1) - 0.5), (idx_j/(W-1) - 0.5)
    grid0 = torch.stack([idx_j.expand([H,-1]), idx_i.expand([-1,W])],-1).unsqueeze(0)

    if W >= H: rh,rw = H/W,1
    else: rh,rw = 1,W/H
    if mode == 'val':
        lh,sh = rh*torch.ones(B, device=device),(0.5-rh/2)*torch.ones(B, device=device)
        lw,sw = rw*torch.ones(B, device=device),(0.5-rw/2)*torch.ones(B, device=device)
    else:
        lh,lw = (0.25+0.75*torch.rand(B, device=device))*rh,(0.25+0.75*torch.rand(B, device=device))*rw
        sh,sw = (1.0 - lh)*torch.rand(B, device=device), (1.0 - lw)*torch.rand(B, device=device)

    norms = []
    for i in range(B):
        norm = norm_s[i, :, :, int(h*sh[i]):int(h*(sh[i] + lh[i])), int(w*sw[i]):int(w*(sw[i] + lw[i]))]
        norm = F.interpolate(norm, size=(H,W),mode='bilinear')
        norms.append(norm)
    norm_s = torch.stack(norms,0).permute(0,1,3,4,2)
    norm_s = F.normalize(norm_s,dim=-1).flatten(0,1)

    s = scale[0] + (scale[1] - scale[0])*torch.rand(1, device=device)
    d = depth[0] + (depth[1] - depth[0])*torch.rand(1, device=device)
    n0 = torch.zeros(1,H,W,3, device=device)
    n0[:,:,:,2] = 1
    v = get_refraction(n0,-(norm_s*s+n0*(1 - s)), n=n)[:,:,:,:2]
    grid = 2.0*(v*d + grid0) #-1,1 range
    if not x2_sample:
        deformed = F.grid_sample(x.expand(-1,L,-1,-1,-1).flatten(0,1), grid,
                                        mode='bicubic', padding_mode='zeros', align_corners=True)
    else:
        grid = F.interpolate(grid.permute(0, 3, 1, 2), scale_factor=2, mode='bilinear').permute(0, 2, 3, 1)
        deformed = F.grid_sample(x.expand(-1,L,-1,-1,-1).flatten(0,1), grid,
                                            mode='bicubic', padding_mode='zeros', align_corners=True)
        deformed = F.interpolate(deformed, scale_factor=0.5, mode='bilinear', antialias=True)
    return torch.nan_to_num(deformed.view(B,L,-1,H,W)).clip(0,1)

class SEALS_Dataset(Dataset):
    def __init__(self, 
                 path_b='/data/sandbox/mshugaev/SEALS/data/synthetic_data_cvpr_092225/backgrounds', 
                 path_w='/data/sandbox/mshugaev/SEALS/data/synthetic_data_cvpr_092225/wave_profiles/Ocean_waves',
                 mode='val', sz=(512,512), L=49, Lmax=200, depth=1.0, scale=1.0, n=1.33, Nmax=-1, stride=1,
                 crop_wave=True, model_path = "THUDM/CogVideoX-2b", Ltext=226, fast=False, 
                 val_subset=True, x2_sample=False,
                 **kwargs):
        #assert stride > 0 and stride <= Lmax//L, 'incorrect stride'
        self.path_b, self.path_w = path_b, path_w
        self.mode = mode
        self.sz,self.L = sz,L
        self.stride = stride
        self.Lmax = Lmax
        if not isinstance(scale,list) and not isinstance(scale,set): scale = (scale,scale)
        self.depth,self.scale = depth,scale
        self.n = n
        self.Nmax = Nmax if mode == 'train' else -1
        self.crop_wave = crop_wave
        
        self.backgrounds = [os.path.join(path_b,f) for f in sorted(os.listdir(path_b)) if f.split('.')[-1] != 'pickle']
        self.waves = [os.path.join(path_w,f) for f in sorted(os.listdir(path_w))]

        self.ids = range(len(self.backgrounds)*len(self.waves))
        if val_subset and os.path.exists(os.path.join(path_b, 'subset.pickle')):
            with open(os.path.join(path_b, 'subset.pickle'), 'rb') as f: self.ids = pickle.load(f)
        #if val_subset > 0 and mode == 'val': #sample uniformly a subset of waves and backgrounds
        #    b = np.arange(val_subset)%len(self.backgrounds)
        #    w = np.arange(val_subset)%len(self.waves)
        #    np.random.seed(43)
        #    np.random.shuffle(b)
        #    np.random.shuffle(w)
        #    self.ids = sorted(b*len(self.waves) + w)
        self.length = len(self.ids)
        self.aug = aug_train(sz) if mode == 'train' else aug_val(sz)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        with open(os.path.join(path_b, 'captions.pickle'), 'rb') as f: self.captions = pickle.load(f)
        self.Ltext = Ltext
        self.fast = fast
        self.x2_sample = x2_sample
        
    def __len__(self):
        return self.length if self.Nmax == -1 else self.Nmax
    
    def __getitem__(self, idx):
        while True:
            if self.Nmax != -1: idx = random.randrange(self.length)
            idx = self.ids[idx]  
            idx_wave = idx%len(self.waves)
            idx_background = idx//len(self.waves)
            #print(idx,idx_wave,idx_background)
            
            try:
                img = load_PIL(self.backgrounds[idx_background])
                break
            except Exception as e:
                idx = random.randrange(self.length)
                continue
            
        img = self.aug(image=img)['image']
        x = torch.from_numpy(img.astype(np.float32)/255).permute(2,0,1)

        files = [os.path.join(self.waves[idx_wave],p) for p in sorted(os.listdir(self.waves[idx_wave]),
                                                                      key=lambda x: int(x.split('_')[-1].split('.')[0]))]
        
        if isinstance(self.stride,int): stride = self.stride
        else: stride = random.randint(self.stride[0], self.stride[1])
        assert len(files) >= self.L*stride, 'Too long sequence'
        if self.mode == 'val':
            ids = range(0, self.L*stride, stride)
        else:
            s = random.randrange(0,len(files)-self.L*stride)
            ids = range(s,s+self.L*stride, stride)
        
        norm_s = []
        for k in ids: norm_s.append(np.load(files[k]))
        norm_s = torch.from_numpy(np.stack(norm_s,0)).float()
        if self.mode == 'val':
            flip_h,flip_w,flip_hw = False,False,False
        else:
            flip_h,flip_w,flip_hw = random.random() > 0.5, random.random() > 0.5, random.random() > 0.5
        if flip_h: norm_s = norm_s.flip(1)
        if flip_w: norm_s = norm_s.flip(2)
        if flip_hw: norm_s = norm_s.permute(0,2,1,3)

        k = '/'.join(self.backgrounds[idx_background].split('/')[-1:])
        caption = self.captions[k] if k in self.captions else ''
        if k not in self.captions: print(k)
        if self.mode == 'train' and random.random() < 0.2: caption = ''
        ids = self.tokenizer(
            caption,
            padding="max_length",
            max_length=self.Ltext,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )['input_ids'].squeeze(0)
        
        if self.fast:
            return {'video': x.unsqueeze(0), 
                'norms': norm_s.permute(0,3,1,2),
                'prompt':ids,
               }  
        else:
            deformed = gen_video(x.unsqueeze(0).unsqueeze(0), norm_s.permute(0,3,1,2).unsqueeze(0), 
                             scale=self.scale, depth=self.depth, mode=self.mode, x2_sample=self.x2_sample)[0]
        
            return {'video': x.unsqueeze(0), 
                'ref_video': deformed,
                'prompt':ids,
               }   
               
class SEALS_Dataset_gt(Dataset):
    def __init__(self, 
                 path_b='/data/sandbox/mshugaev/SEALS/data/synthetic_data_cvpr_092225/backgrounds', 
                 path_w='/data/sandbox/mshugaev/SEALS/data/synthetic_data_cvpr_092225/wave_profiles/Ocean_waves',
                 mode='val', sz=(512,512), L=49, Lmax=200, depth=1.0, scale=1.0, n=1.33, Nmax=-1, stride=1,
                 crop_wave=True, model_path = "THUDM/CogVideoX-2b", Ltext=226, fast=False, 
                 val_subset=True,
                 **kwargs):
        #assert stride > 0 and stride <= Lmax//L, 'incorrect stride'
        self.path_b, self.path_w = path_b, path_w
        self.mode = mode
        self.sz,self.L = sz,L
        self.stride = stride
        self.Lmax = Lmax
        if not isinstance(scale,list) and not isinstance(scale,set): scale = (scale,scale)
        self.depth,self.scale = depth,scale
        self.n = n
        self.Nmax = Nmax if mode == 'train' else -1
        self.crop_wave = crop_wave
        
        self.backgrounds = [os.path.join(path_b,f) for f in sorted(os.listdir(path_b)) if f.split('.')[-1] != 'pickle']
        self.waves = [os.path.join(path_w,f) for f in sorted(os.listdir(path_w))]

        self.ids = range(len(self.backgrounds)*len(self.waves))
        if val_subset and os.path.exists(os.path.join(path_b, 'subset.pickle')):
            with open(os.path.join(path_b, 'subset.pickle'), 'rb') as f: self.ids = pickle.load(f)
        #if val_subset > 0 and mode == 'val': #sample uniformly a subset of waves and backgrounds
        #    b = np.arange(val_subset)%len(self.backgrounds)
        #    w = np.arange(val_subset)%len(self.waves)
        #    np.random.seed(43)
        #    np.random.shuffle(b)
        #    np.random.shuffle(w)
        #    self.ids = sorted(b*len(self.waves) + w)
        self.length = len(self.ids)
        
    def __len__(self):
        return self.length if self.Nmax == -1 else self.Nmax
    
    def __getitem__(self, idx):
        if self.Nmax != -1: idx = random.randrange(self.length)
        idx = self.ids[idx]  
        idx_wave = idx%len(self.waves)
        idx_background = idx//len(self.waves)

        return self.backgrounds[idx_background]