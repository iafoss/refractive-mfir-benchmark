import warnings
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as ssim

from utils_unsupervised_ndir.common_utils import *
from networks_unsupervised_ndir.conv_layers import *


#### Required functions 

def backwarp(tenInput, tenFlow):
    backwarp_tenGrid = {}
    if str(tenFlow.size()) not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])

        backwarp_tenGrid[str(tenFlow.size())] = torch.cat([ tenHorizontal, tenVertical ], 1).cuda()
    # end

    tenFlow = torch.cat([
        tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
        tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)
    ], 1)

    return torch.nn.functional.grid_sample(
        input=tenInput,
        grid=(backwarp_tenGrid[str(tenFlow.size())] + tenFlow).permute(0, 2, 3, 1),
        mode='bilinear',
        padding_mode='zeros'
    )

def backwarp_grid(tenInput, tenFlow_xy):
    return torch.nn.functional.grid_sample(
        input=tenInput,
        grid=tenFlow_xy.permute(0, 2, 3, 1),
        mode='bilinear',
        padding_mode='zeros'
    )

def im_resize(im, scale_factor):
    width = int(im.size[1] * scale_factor)
    height = int(im.size[0] * scale_factor)
    newsize = (height, width)
    return im.resize(newsize)

def visualize_rgb(warp_np):
    nr = warp_np.shape[0]
    nc = warp_np.shape[1]
    warp_np = (warp_np - np.amin(warp_np)) / (np.amax(warp_np) - np.amin(warp_np))
    one_pad = np.ones((nr, nc, 1))
    out_warp_np = np.concatenate((warp_np, one_pad), axis=-1)
    return out_warp_np

def visualize_rgb_norm(warp_np):
    nr = warp_np.shape[0]
    nc = warp_np.shape[1]
    one_pad = np.ones((nr, nc, 1))
    out_warp_np = np.concatenate((warp_np, one_pad), axis=-1)
    return out_warp_np

def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def to_uint8(image):
    if image.ndim == 3 and image.shape[0] in [1, 3]:
        image = image.transpose(1, 2, 0)
    if image.dtype in [np.float32, np.float64]:
        image = np.clip(image, 0, 1)
        image = (image * 255).astype(np.uint8)
    return image


#### Setup Fourier Feature Transform function ####

class GaussianFourierFeatureTransform_B(torch.nn.Module):
    def __init__(self, num_input_channels, B, mapping_size=256, scale=10):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self._B = B * scale

    def forward(self, x):
        assert x.dim() == 4, 'Expected 4D input (got {}D input)'.format(x.dim())

        batches, channels, width, height = x.shape

        assert channels == self._num_input_channels, \
            "Expected input to have {} channels (got {} channels)".format(
                self._num_input_channels, channels
            )

        x = x.permute(0, 2, 3, 1).reshape(batches * width * height, channels)
        x = x @ self._B.to(x.device)
        x = x.view(batches, width, height, self._mapping_size)
        x = x.permute(0, 3, 1, 2)

        x = 2 * np.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

warnings.filterwarnings("ignore")

imsize = -1
extensions = ['.jpg', '.JPG', '.png', '.ppm', '.bmp', '.pgm', '.tif']


class UnsupervisedNDIRModel:
    """
    256-dataset version aligned with the working defor7_11.py flow.

    Optimization frames are selected like:
        start_frame = 1
        num_samples = 11
        frame_step = 2
        frame_indices = np.arange(start_frame, start_frame + frame_step * num_samples, frame_step)

    Example selected indices:
        [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]

    Input:
        [L, 3, H, W]
        or [B, L, 3, H, W]

    Output:
        if output_mode='single'   -> [B, 1, 3, H, W]
        if output_mode='sequence' -> [B, L_out, 3, H, W]

    This version follows defor7_11.py:
        - grid is built directly at the working image size
        - no forced 512 upsample
        - no scale_factor=2 upsample on refined_xy
    """
    def __init__(
        self,
        batch_size=11,
        scale_factor=1,
        num_iter_i=1000,
        num_iter=1000,
        FB_img=8,
        vec_scale=1.1,
        start_frame=1,
        wave_cycle=21.4,
        num_samples=11,
        frame_step=2,
        sz=256,
        output_mode='single',
        device='cuda'
    ):
        self.batch_size = batch_size
        self.scale_factor = scale_factor
        self.num_iter_i = num_iter_i
        self.num_iter = num_iter
        self.FB_img = FB_img
        self.vec_scale = vec_scale

        self.start_frame = start_frame
        self.wave_cycle = wave_cycle
        self.num_samples = num_samples
        self.frame_step = frame_step
        self.sz = sz

        self.output_mode = output_mode
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.cuda.FloatTensor if self.device.type == 'cuda' else torch.FloatTensor

        if self.batch_size != self.num_samples:
            raise ValueError("For original-style behavior, batch_size and num_samples should be the same.")

    def _optimize_and_predict_single_video(self, video):
        if video.dim() != 4:
            raise ValueError(f'Expected [L,3,H,W], got {tuple(video.shape)}')

        total_imgs = video.shape[0]

        # minimal change: always run internally at 256x256
        sz0 = video.shape[-2:]
        if video.shape[-2:] != (self.sz, self.sz):
            video = F.interpolate(
                video.float(),
                size=(self.sz, self.sz),
                #mode='area',
                mode='bilinear',
                align_corners=False
            )

        frame_indices = np.arange(
            self.start_frame,
            self.start_frame + self.frame_step * self.num_samples,
            self.frame_step
        )

        if total_imgs <= frame_indices[-1]:
            raise ValueError(
                f"Input sequence has {total_imgs} frames, but interval-{self.frame_step} selection needs frame index "
                f"{frame_indices[-1]}. Provide at least {frame_indices[-1] + 1} frames."
            )

        images_warp_np = video.detach().cpu().numpy()
        images_mean_np = np.mean(images_warp_np, axis=0)
        dim, nr, nc = images_mean_np.shape

        start_frame = self.start_frame
        wave_cycle = self.wave_cycle
        num_samples = self.num_samples
        frame_step = self.frame_step

        frame_indices = np.arange(
            start_frame,
            start_frame + frame_step * num_samples,
            frame_step
        )
        selected_frames = images_warp_np[frame_indices]

        # Build the grid directly at the image size, like defor7_11.py
        coords_x = np.linspace(-1, 1, nc)
        coords_y = np.linspace(-1, 1, nr)
        xy_grid = np.stack(np.meshgrid(coords_x, coords_y), -1)

        xy_grid_var = np_to_torch(xy_grid.transpose(2, 0, 1)).type(self.dtype).to(self.device)
        xy_grid_batch_var = xy_grid_var.repeat(self.batch_size, 1, 1, 1)

        model_imgen = conv_layers(256, 3)
        model_imgen = model_imgen.type(self.dtype).to(self.device)

        torch.manual_seed(0)
        B_var = torch.randn(2, 128, device=self.device)

        model_grid = []
        for i in range(self.batch_size):
            model_grid.append(conv_layers(2, 2, need_sigmoid=False, need_tanh=True).to(self.device))

        img_gt_batch_var = torch.from_numpy(selected_frames).type(self.dtype).to(self.device)

        FB_img = self.FB_img
        vec_scale = self.vec_scale

        straight_grid_input = GaussianFourierFeatureTransform_B(2, B_var, 128, FB_img)(xy_grid_batch_var)
        grid_input_single_gd = xy_grid_var.detach().clone()
        grid_input_gd = xy_grid_batch_var.detach().clone()
        grid_input = straight_grid_input.detach().clone()

        model_params_list = [{'params': model_grid[i].parameters()} for i in range(self.batch_size)]
        model_params_list.append({'params': model_imgen.parameters()})
        optimizer = torch.optim.Adam(model_params_list, lr=1e-4)

        # 1st optimization -- same style as defor7_11.py, no forced upscale
        for epoch in range(self.num_iter_i):
            optimizer.zero_grad()

            refined_xy = []
            for b in range(self.batch_size):
                vec_input = grid_input_single_gd
                refined_xy.append(model_grid[b](vec_input))

            refined_xy = vec_scale * torch.cat(refined_xy)
            generated = model_imgen(grid_input)

            loss = torch.nn.functional.l1_loss(img_gt_batch_var, generated)
            loss += torch.nn.functional.l1_loss(xy_grid_batch_var, refined_xy)

            loss.backward()
            optimizer.step()

        img_gt_np = images_mean_np.clip(0, 1)
        i = 0

        loss_arr = torch.zeros(self.num_iter)
        psnr_arr_sharp = torch.zeros(self.num_iter)
        psnr_arr_turb = torch.zeros(self.num_iter)
        ssim_arr_sharp = torch.zeros(self.num_iter)
        ssim_arr_turb = torch.zeros(self.num_iter)

        optimizer = torch.optim.Adam(model_params_list, lr=1e-4)

        for epoch in range(self.num_iter):
            optimizer.zero_grad()

            refined_xy = []
            for b in range(self.batch_size):
                vec_input = grid_input_single_gd
                refined_xy.append(model_grid[b](vec_input))

            refined_xy = vec_scale * torch.cat(refined_xy)

            refined_warp = refined_xy - xy_grid_batch_var
            refined_uv = torch.cat(
                (
                    (nc - 1.0) * refined_warp[:, 0:1, :, :] / 2,
                    (nr - 1.0) * refined_warp[:, 1:2, :, :] / 2
                ),
                1
            )

            mask_u1 = (refined_xy[:, 0:1, :, :] > -1).float() * 1
            mask_u2 = (refined_xy[:, 0:1, :, :] < 1).float() * 1
            mask_v1 = (refined_xy[:, 1:2, :, :] > -1).float() * 1
            mask_v2 = (refined_xy[:, 1:2, :, :] < 1).float() * 1
            mask = mask_u1 * mask_u2 * mask_v1 * mask_v2

            sharp_imgs_predict = model_imgen(grid_input)
            refined_turb_imgs = backwarp_grid(sharp_imgs_predict, refined_xy)
            generated_turb_imgs = model_imgen(
                GaussianFourierFeatureTransform_B(2, B_var, 128, FB_img)(refined_xy)
            )

            loss = torch.nn.functional.l1_loss(generated_turb_imgs * mask, img_gt_batch_var * mask)
            loss += torch.nn.functional.l1_loss(refined_turb_imgs * mask, img_gt_batch_var * mask)
            loss += torch.nn.functional.l1_loss(generated_turb_imgs * mask, refined_turb_imgs * mask)

            loss_arr[epoch] = loss.detach().cpu()

            try:
                psnr_arr_sharp[epoch] = compare_psnr(img_gt_np, sharp_imgs_predict[i].detach().cpu().numpy())
                psnr_arr_turb[epoch] = compare_psnr(images_warp_np[i], generated_turb_imgs[i].detach().cpu().numpy())
                ssim_arr_sharp[epoch] = float(
                    ssim(
                        img_gt_np.transpose(1, 2, 0),
                        sharp_imgs_predict[i].detach().cpu().numpy().transpose(1, 2, 0),
                        data_range=1.0,
                        channel_axis=2
                    )
                )
                ssim_arr_turb[epoch] = float(
                    ssim(
                        images_warp_np[i].transpose(1, 2, 0),
                        generated_turb_imgs[i].detach().cpu().numpy().transpose(1, 2, 0),
                        data_range=1.0,
                        channel_axis=2
                    )
                )
            except Exception:
                pass

            loss.backward()
            optimizer.step()

        # Test
        segments = int(total_imgs / self.batch_size)
        pred_list = []

        for i in range(segments):
            if i * self.batch_size > total_imgs:
                break

            img_gt_batch_var = torch.from_numpy(
                images_warp_np[i * self.batch_size:(i + 1) * self.batch_size]
            ).type(self.dtype).to(self.device)

            straight_grid_input = GaussianFourierFeatureTransform_B(2, B_var, 128, FB_img)(xy_grid_batch_var)
            grid_input_single_gd = xy_grid_var.detach().clone()
            grid_input_gd = xy_grid_batch_var.detach().clone()
            grid_input = straight_grid_input.detach().clone()

            refined_xy = []
            for b in range(self.batch_size):
                refined_xy.append(model_grid[b](grid_input_single_gd))

            refined_xy = vec_scale * torch.cat(refined_xy)

            sharp_imgs_predict = model_imgen(grid_input)
            pred_list.append(sharp_imgs_predict.detach().cpu().clip(0, 1))

        if len(pred_list) == 0:
            raise ValueError("No full batch segment found in input sequence.")

        pred_seq = torch.cat(pred_list, dim=0)
        
        if pred_seq.shape[-2:] != sz0:
            pred_seq = F.interpolate(pred_seq, size=sz0, mode='bicubic')

        if self.output_mode == 'single':
            return pred_seq[0:1]
        elif self.output_mode == 'sequence':
            return pred_seq
        else:
            raise ValueError(f"Unknown output_mode: {self.output_mode}")

    def __call__(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(0)
        elif x.dim() != 5:
            raise ValueError(f'Expected [L,3,H,W] or [B,L,3,H,W], got {tuple(x.shape)}')

        outputs = []
        for b in range(x.shape[0]):
            outputs.append(self._optimize_and_predict_single_video(x[b]))

        return torch.stack(outputs, dim=0)
