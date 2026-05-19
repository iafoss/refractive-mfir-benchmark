import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from collections import OrderedDict
from unsupervised_ndir_model_original_style import UnsupervisedNDIRModel

def gen_img_ref(x0,k=None):
    return (x0['ref_video']*255).byte()
    
def gen_img_mean(x0,k=None):
    return (255*x0['ref_video'].mean(0, keepdim=True)).byte()
    
from DATUM import Model as DATUM_Model
class gen_DATUM:
    def __init__(self, spynet_path='DATUM/spynet_init.pth', datum_path='DATUM/DATUM_static.pth', device='cuda', **kwargs):
        config = {
        'model': 'DATUM',
        'n_features': 16,
        'n_blocks': 15,
        'future_frames': 2,
        'past_frames': 2,
        'activation': 'gelu',
        'spynet_path': spynet_path,
        'output_full': True,
        }
     
        model = DATUM_Model(**config).to(device)

        sd0 = torch.load(datum_path)
        sd = OrderedDict()
        for k in sd0:
            sd[k.replace('model.model.','model.')] = sd0[k]
        model.load_state_dict(sd)
        model.eval();
        self.model = model
        self.device = device

    def __call__(self, x0,k=None):
        with torch.no_grad():
            v = x0['ref_video'].unsqueeze(0).to(self.device)
            p = self.model({'imgs':v,'tilt':v})
        pred = p['pred'].clip(0,1)[0]
            
        return (pred*255).byte()

import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from functools import partial
import math

def to_single_channel(x, scale=2):
    x = torchvision.transforms.functional.rgb_to_grayscale(x)
    if scale > 1: x = F.avg_pool2d(x, scale)
    return x

def project_image(x, grid):
    grid = F.interpolate(grid, x.shape[-2:], mode='bilinear', align_corners=True)
    p = F.grid_sample(x, grid.permute(0,2,3,1), padding_mode='border', mode='bicubic', align_corners=False)
    return p

class Registrator(nn.Module):
    def __init__(self, sz, N, grid_r=32):
        super().__init__()
        grid = torch.stack([
            torch.arange(-1,1.001,2*grid_r/(sz[1]-grid_r)).unsqueeze(0).expand(sz[0]//grid_r,-1),
            torch.arange(-1,1.001,2*grid_r/(sz[0]-grid_r)).unsqueeze(1).expand(-1,sz[1]//grid_r),
        ], 0)
        self.grid = nn.Parameter(grid.clone().unsqueeze(0).repeat(N,1,1,1))
        self.grid_ref = nn.Parameter(grid.clone().unsqueeze(0), requires_grad=False)

    def forward(self, x):
        x = project_image(x, self.grid)
        return x, self.grid, self.grid_ref
    
    def project(self, x, idx=0):
        with torch.no_grad():
            p = project_image(x.unsqueeze(0), self.grid[idx].unsqueeze(0))[0]
        return p 

    def project_all(self, x):
        with torch.no_grad():
            p = project_image(x, self.grid)
        return p 
    
def loss_tot(x, y, pyramid=3, pyramid_start=0, w_pos=0.05, w_def=2.0, w_def_t=2.0, Nmax=10, th_background=0.02,
            erosion=4):
    x, g, g0 = x
    x0 = y
    N,C,H,W = x.shape
    h,w = g.shape[-2:]
    l = math.sqrt((H*W)/(h*w))
    if erosion > 0: 
        w_erosion = (torch.FloatTensor([[[[0,1,0],[1,1,1],[0,1,0]]]])/5.0).to(x.device)

    loss_m = 0
    # image matching loss
    m  = (x0 > th_background).max(1,keepdim=True)[0].float() # remove effects of balck empty areas
    for k in range(pyramid_start+1,pyramid+1):
        xk = F.avg_pool2d(x, 2**k)
        mk = F.max_pool2d(1-m, 2**k) < 0.5
        if erosion > 0: 
            mk = mk.float()
            for i in range(erosion): mk = F.conv2d(mk,w_erosion,padding=1) #erosion
            mk = mk > 0.95
        
        for i in range(N):
            selection = torch.ones(N, device = x.device, dtype=torch.bool)
            selection[i] = False
            selection = torch.nonzero(selection).view(-1)
            selection = selection[torch.randperm(len(selection))][:Nmax]
            
            x1 = xk[i].unsqueeze(0).repeat(min(N-1,Nmax),1,1,1)
            x2 = xk[selection]
            m1 = mk[i].unsqueeze(0).repeat(min(N-1,Nmax),1,1,1)
            m2 = mk[selection]
            
            m_tot = (m1 & m2).expand(-1,C,-1,-1)
            if m_tot.any():
                loss_m += torch.sqrt(torch.pow(x1[m_tot].view(-1)-x2[m_tot].view(-1), 2) + 1e-4).mean()
    loss_m /= pyramid*N
    
    # displacement penalty
    loss_pos = w_pos*l*F.l1_loss(g.view(-1),g0.repeat(N,1,1,1).view(-1))
    
    # deformation penalty
    loss_def = w_def*torch.cat([
        torch.pow(l*((g[:,:,:,1:] - g[:,:,:,:-1]) - \
        (g0[:,:,:,1:] - g0[:,:,:,:-1])).abs(),2.0).sum(1).view(-1),
        torch.pow(l*((g[:,:,1:,:] - g[:,:,:-1,:]) - \
        (g0[:,:,1:,:] - g0[:,:,:-1,:])).abs(),2.0).sum(1).view(-1),
        torch.pow(l*((g[:,:,1:,1:] - g[:,:,:-1,:-1]) - \
        (g0[:,:,1:,1:] - g0[:,:,:-1,:-1])).abs(),2.0).sum(1).view(-1),
        torch.pow(l*((g[:,:,-1:,1:] - g[:,:,1:,:-1]) - \
        (g0[:,:,-1:,1:] - g0[:,:,1:,:-1])).abs(),2.0).sum(1).view(-1),
    ]).mean()
    loss_def_t = w_def*torch.pow(l*(g[1:] - g[:-1]),2).sum(1).view(-1).mean()
    
    return loss_m + loss_pos + loss_def + loss_def_t

class gen_Grid_registration:
    def __init__(self, 
        img_scale = 2,          # image downscale in registration
        grid_r = 16,            # reduction of grid size vs pixel size
        Nmax = 12,              # maximum number of neighbors
        pyramid = 5,            # number of loss scales in the pyramid
        lr = 1e-2,              # learning rate
        pyramid_start = 0,      # start pyramid from particular scale
        w_pos = 1e-8,           # position penalty weight
        w_def = (5e-2,1e-4),    # deformation penalty weight
        w_def_t = 1e-4,
        th_background = 0.02,   # threshold for empty image parts
        Nsteps = 50,            # number of optimization steps
        erosion = 4,            # mask erosion
        device='cuda', **kwargs):
        
        self.img_scale = img_scale
        self.grid_r = grid_r
        self.Nmax = Nmax
        self.pyramid = pyramid
        self.lr = lr
        self.pyramid_start = pyramid_start
        self.w_pos = w_pos
        self.w_def = w_def
        self.w_def_t = w_def_t
        self.th_background = th_background
        self.Nsteps = Nsteps
        self.erosion = erosion
        self.device = device

    def __call__(self, x0,k=None):
        x = to_single_channel(x0['ref_video'], self.img_scale)

        x = x.to(self.device)
        registrator = Registrator(x.shape[-2:], x.shape[0], grid_r=self.grid_r).to(self.device)
        opt = optim.RAdam(registrator.parameters(), lr=self.lr)
        torch.cuda.empty_cache()
        
        loss_func = partial(loss_tot, pyramid=self.pyramid, pyramid_start=self.pyramid_start,
                          w_pos=self.w_pos, w_def=self.w_def[0], w_def_t=self.w_def_t, Nmax=self.Nmax,
                          th_background=self.th_background, erosion=self.erosion)

        #for k in tqdm(range(Nsteps)):
        for k in range(self.Nsteps):
            loss_func = partial(loss_tot, pyramid=self.pyramid, pyramid_start=self.pyramid_start,
                          w_pos=self.w_pos, w_def=self.w_def[0] - (self.w_def[0] - self.w_def[1])*(k/(self.Nsteps-1))**0.5,
                          w_def_t=self.w_def_t, Nmax=self.Nmax, th_background=self.th_background, erosion=self.erosion)

            opt.zero_grad()
            p = registrator(x)
            loss = loss_func(p,x)
            loss.backward()
            opt.step()
            #if k%10 == 0 or k == Nsteps-1: print(k, loss.detach().item())
        
        torch.cuda.empty_cache() 

        xp = registrator.project_all(x0['ref_video']).clip(0,1)
        return (xp*255).byte()
        
class gen_FILE:
    def __init__(self, path, device='cuda', ext='png', **kwargs):
        self.path, self.ext = path, ext
        self.device = device

    def __call__(self, x0, k):
        pred = cv2.imread(os.path.join(self.path,f'{k}.{self.ext}'))[:,:,::-1].copy()
        pred = torch.from_numpy(pred).permute(2,0,1).unsqueeze(0).to(self.device)
            
        return pred
        
class gen_VIDEO:
    def __init__(self, path, device='cuda', ext='avi', L=1, **kwargs):
        self.path, self.ext = path, ext
        self.L = L
        self.device = device

    def __call__(self, x0, k):
        video = cv2.VideoCapture(os.path.join(self.path,f'{k}.{self.ext}'))
        pred = []
        for k in range(self.L):
            x = video.read()
            if not x[0]: break
            pred.append(torch.from_numpy(x[1][:,:,::-1].copy()).permute(2,0,1))
        video.release()
        pred = torch.stack(pred,0).to(self.device)
            
        return pred
        

class gen_UNSUPERVISED_NDIR:
    def __init__(
        self,
        batch_size=10,
        scale_factor=1,
        num_iter_i=1000,
        num_iter=1000,
        FB_img=8,
        vec_scale=1.1,
        start_frame=1,
        wave_cycle=21.4,
        num_samples=10,
        output_mode='sequence',
        device='cuda',
        **kwargs
    ):
        self.model = UnsupervisedNDIRModel(
            batch_size=batch_size,
            scale_factor=scale_factor,
            num_iter_i=num_iter_i,
            num_iter=num_iter,
            FB_img=FB_img,
            vec_scale=vec_scale,
            start_frame=start_frame,
            wave_cycle=wave_cycle,
            num_samples=num_samples,
            output_mode=output_mode,
            device=device
        )
        self.device = device

    def __call__(self, x0, k=None):
        ref_video = x0['ref_video'].float().to(self.device)

        if ref_video.max() > 1.0:
            ref_video = ref_video / 255.0

        pred = self.model(ref_video).clip(0,1)[0] #Lx3xHxW
        return (pred * 255).byte()
    