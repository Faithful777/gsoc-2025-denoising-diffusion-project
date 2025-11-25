#module denoising_diffusion.py

import os
import time
import glob
import numpy as np
import tqdm
import torch
import torch.nn as nn
from tqdm import tqdm as tm
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt

import torch
from torchmetrics.functional import structural_similarity_index_measure as ssim

import torch.fft as fft
from torch.nn import functional as F
from diffusion_unet import DiffusionUNet
from utils import get_optimizer, save_checkpoint, load_checkpoint, generalized_steps, generalized_steps_overlapping


# This script is adapted from the following repositories
# https://github.com/ermongroup/ddim
# https://github.com/bahjat-kawar/ddrm


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

class FourierLoss(nn.Module):  # nn.Module
    def __init__(self, loss_weight=1.0, reduction='mean', **kwargs):
        super(FourierLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, sr, hr):
        sr_w, hr_w = self.addWindows(sr, hr)
        sr_mag, sr_ang = self.comFourier(sr_w)
        hr_mag, hr_ang = self.comFourier(hr_w)
        mag_loss = self.get_l1loss(sr_mag, hr_mag, weight=self.loss_weight, reduction=self.reduction)
        ang_loss = self.get_angleloss(sr_ang, hr_ang, weight=self.loss_weight, reduction=self.reduction)
        # ang_loss = self.get_l1loss(sr_ang, hr_ang, weight=self.loss_weight, reduction=self.reduction)
        return (mag_loss + ang_loss)

    def comFourier(self, image):
        B, C, H, W = image.shape
        frm_list = []
        fre_quen = []
        for i in range(C):
            in_ = image[:, i:i + 1, :, :]
            fftn = fft.fftn(in_, dim=(2, 3))
            # add shift
            fftn_shift = fft.fftshift(fftn)  # + 1e-8
            # print('fftn:', fftn_shift.size())
            frm_list.append(fftn_shift.real)
            fre_quen.append(fftn_shift.imag)
        fre_mag = torch.cat(frm_list, dim=1)
        fre_ang = torch.cat(fre_quen, dim=1)

        return fre_mag, fre_ang

    def addWindows(self, sr, hr):
        b, c, h, w = sr.size()
        win1 = torch.hann_window(h).reshape(h, 1).to(sr.device)
        win2 = torch.hann_window(w).reshape(1, w).to(sr.device)
        win = torch.mm(win1, win2)
        sr, hr = sr * win, hr * win
        return sr, hr

    def get_angleloss(self, pred, target, weight, reduction):
        # minimum = torch.minimum(pred, target)
        diff = pred - target
        minimum = torch.min(diff)
        loss = torch.mean(torch.abs(minimum))
        return weight * loss

    def get_l1loss(self, pred, target, weight, reduction):
        loss = F.mse_loss(pred, target, reduction=reduction)
        return weight * loss



def ssim_noise_estimation_loss(model, x0, t, e, b, alpha_schedule=None, ssim_weight=1.0):
    """
    model        : diffusion model
    x0           : full input (includes conditioning + target clean image)
    t            : timesteps
    e            : sampled Gaussian noise
    b            : beta schedule
    ssim_weight  : relative weighting factor for SSIM contribution
    """

    a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    clean_target = x0[:, 2:, :, :]
    x = clean_target * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(torch.cat([x0[:, :2, :, :], x], dim=1), t.float())
    noise_loss = (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)
    x0_hat = (x - output * (1 - a).sqrt()) / a.sqrt()  #Reconstruct clean image from noisy input

    ssim_val = ssim(x0_hat, clean_target, data_range=1.0)  # assumes images in [0,1]
    ssim_loss = 1.0 - ssim_val
    total_loss = noise_loss + ssim_weight * noise_loss.detach() * ssim_loss

    F_loss = FourierLoss()
    f_loss = F_loss.forward(x0_hat, clean_target)
    total_loss = total_loss + 0.05*f_loss

    return total_loss


def noise_estimation_loss(model, x0, t, e, b):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0[:, 2:, :, :] * a.sqrt() + e * (1.0 - a).sqrt()
    #output = model(x, t.float())
    output = model(torch.cat([x0[:, :2, :, :], x], dim=1), t.float())
    return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


class DenoisingDiffusion(object):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device

        self.model = DiffusionUNet(config)
        self.model.to(self.device)
        self.model = torch.nn.DataParallel(self.model)

        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)

        self.optimizer = get_optimizer(self.config, self.model.parameters())
        self.start_epoch, self.step = 0, 0

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

    def load_ddm_ckpt(self, load_path, ema=False):
        checkpoint = load_checkpoint(load_path, None)
        self.start_epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.ema_helper.load_state_dict(checkpoint['ema_helper'])
        if ema:
            self.ema_helper.ema(self.model)
        print("=> loaded checkpoint '{}' (epoch {}, step {})".format(load_path, checkpoint['epoch'], self.step))

    def train(self, DATASET):
        cudnn.benchmark = True
        train_loader, val_loader = DATASET.get_loaders()

        if os.path.isfile(self.args.resume):
            self.load_ddm_ckpt(self.args.resume)

        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            print('epoch: ', epoch)
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                n = x.size(0)
                data_time += time.time() - data_start
                self.model.train()
                self.step += 1

                x = x.to(self.device)
                #x = data_transform(x)
                e = torch.randn_like(x[:, 2:, :, :]) * 0.15
                b = self.betas

                # antithetic sampling
                t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                loss = ssim_noise_estimation_loss(self.model, x, t, e, b)
                #loss = weighted_noise_estimation_loss(self.model, x, t, e, b)

                if self.step % 100 == 0:
                    print(f"step: {self.step}, loss: {loss.item()}, data time: {data_time / (i+1)}")

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema_helper.update(self.model)
                data_start = time.time()

                if self.step % self.config.training.validation_freq == 0:
                    self.model.eval()
                    self.sample_validation_patches(val_loader, self.step)

                if self.step % self.config.training.snapshot_freq == 0 or self.step == 1:
                    print(f"Saving checkpoint at step: {self.step}")
                    print(os.path.join(self.config.data.data_dir, 'ckpts', self.config.data.dataset + '_ddpm'))
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'step': self.step,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'ema_helper': self.ema_helper.state_dict(),
                        'params': self.args,
                        'config': self.config
                    }, filename=os.path.join(self.config.data.data_dir, 'ckpts', self.config.data.dataset + '_ddpm'))

    def sample_image(self, x_cond, x, latent=None, last=True, patch_locs=None, patch_size=None):
        skip = self.config.diffusion.num_diffusion_timesteps // self.args.sampling_timesteps
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)
        if patch_locs is not None and len(patch_locs) > 1:
            print('Using generalized steps overlapping')
            xs = generalized_steps_overlapping(x, x_cond, seq, self.model, self.betas,latent=latent, eta=0.,
                                                              corners=patch_locs, p_size=patch_size, manual_batching=False)
        else:
            xs = generalized_steps(x, x_cond, seq, self.model, self.betas, eta=0.)
        if last:
            xs = xs[0][-1]
        return xs

    def sample_validation_patches(self, val_loader, step):
        image_folder = os.path.join(self.args.image_folder, self.config.data.dataset + str(self.config.data.image_size))
        with torch.no_grad():
            print(f"Processing a single batch of validation images at step: {step}")
            for i, (x, y) in enumerate(val_loader):
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                break
            n = x.size(0)
            x_cond = x[:, :2, :, :].to(self.device)
            x_targ = x[:, 2:, :, :]
            #x_cond = data_transform(x_cond)
            x = torch.randn(n, 1, self.config.data.image_size, self.config.data.image_size, device=self.device) * 0.15
            x = self.sample_image(x_cond, x)
            #x = inverse_data_transform(x)
            #x_cond = inverse_data_transform(x_cond)

            for i in range(n):
                pass
                #save_image(x_cond[i], os.path.join(image_folder, str(step), f"{i}_cond.png"))
                #save_image(x[i], os.path.join(image_folder, str(step), f"{i}.png"))


            # Plot the images side by side
            fig, axes = plt.subplots(n, 3, figsize=(6, 3*n))
            for i in range(n):
                # Original conditioned image
                axes[i, 0].imshow(x_cond[i][:1,:,:].permute(1, 2, 0).cpu().numpy(), cmap='gray', vmin=0, vmax=1)
                axes[i, 0].axis('off')
                axes[i, 0].set_title(f'Cond Image {i}')
                # Corresponding generated image
                axes[i, 1].imshow(x[i].permute(1, 2, 0).cpu().numpy(), cmap='gray', vmin=0, vmax=1)
                axes[i, 1].axis('off')
                axes[i, 1].set_title(f'Gen Image {i}')
                # Corresponding generated image
                axes[i, 2].imshow(x_targ[i].permute(1, 2, 0).cpu().numpy(), cmap='gray', vmin=0, vmax=1)
                axes[i, 2].axis('off')
                axes[i, 2].set_title(f'Target Image {i}')
            plt.tight_layout()
            plt.show()
