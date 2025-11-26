#module restoration.py

import torch
import torch.nn as nn
#import utils
import torchvision
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from torchmetrics.functional import structural_similarity_index_measure as ssim
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


class DiffusiveRestoration:
    def __init__(self, diffusion, args, config):
        super(DiffusiveRestoration, self).__init__()
        self.args = args
        self.config = config
        self.diffusion = diffusion

        if os.path.isfile(args.resume):
            self.diffusion.load_ddm_ckpt(args.resume, ema=False)
            self.diffusion.model.eval()
        else:
            print('Pre-trained diffusion model path is missing!')


    def restore(self, val_loader, validation='snow', r=None, show_images=False):
        image_folder = os.path.join(self.args.image_folder, self.config.data.dataset, validation)
        os.makedirs(image_folder, exist_ok=True)
    
        all_outputs = []  # <-- store all restored outputs for final save
    
        with torch.no_grad():
            for i, (x, lt, y) in enumerate(tqdm(val_loader)):
                print(f"starting the processing from image {y}")
    
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                x_cond = x[:, :1, :, :].to(self.diffusion.device)
                x_output = self.diffusive_restoration(x_cond, lt, r=r)
    
                # save batch output
                #file_path = os.path.join(image_folder, f"{y}_output.npy")
                #np.save(file_path, x_output.cpu().numpy())
                #save_image(x_output, os.path.join(image_folder, f"{y}_output.png"))
    
                # collect output for final combined save
                all_outputs.append(x_output.cpu().numpy())
    
                # prepare tensors
                images = x_output
                target = x[:, 1:, :, :].to(self.diffusion.device)
    
                print("Shapes:")
                print("cond:", x_cond.shape, "output:", images.shape, "target:", target.shape)
    
                # ======================================================
                # OPTIONAL DISPLAY BLOCK
                # ======================================================
                if show_images:
                    n = x.shape[0]
                    fig, axes = plt.subplots(n, 3, figsize=(12, 4*n))
                    if n == 1:
                        axes = axes[None, :]
    
                    for idx in range(n):
                        axes[idx, 0].imshow(x_cond[idx].squeeze().cpu().numpy(),
                                            cmap='gray', vmin=0, vmax=1)
                        axes[idx, 0].set_title("Conditional Input")
                        axes[idx, 0].axis('off')
    
                        axes[idx, 1].imshow(images[idx].squeeze().cpu().numpy(),
                                            cmap='gray', vmin=0, vmax=1)
                        axes[idx, 1].set_title("Restored Output")
                        axes[idx, 1].axis('off')
    
                        axes[idx, 2].imshow(target[idx].squeeze().cpu().numpy(),
                                            cmap='gray', vmin=0, vmax=1)
                        axes[idx, 2].set_title("Ground Truth")
                        axes[idx, 2].axis('off')
    
                    plt.tight_layout()
                    plt.show()
                # ======================================================
                images = images.to(self.diffusion.device)
                target = target.to(self.diffusion.device)
    
                # compute metrics
                mse_loss = F.mse_loss(images, target).item()
                ssim_score = ssim(images, target, data_range=1.0)
    
                print(f"MSE Loss: {mse_loss:.6f}")
                print(f"SSIM Score: {ssim_score:.6f}")
                #break
    
            # END FOR
    
        # ==========================================================
        # SAVE FULL OUTPUT COLLECTION
        # ==========================================================
        all_outputs = np.concatenate(all_outputs, axis=0)
        final_path = os.path.join(image_folder, f"{validation}.npy")
        np.save(final_path, all_outputs)
    
        print(f"\nSaved combined output array: {final_path}")
        print(f"Shape: {all_outputs.shape}")

    def diffusive_restoration(self, x_cond, lt, r=None):
        p_size = self.config.data.image_size
        h_list, w_list = self.overlapping_grid_indices(x_cond, output_size=p_size, r=r)
        corners = [(i, j) for i in h_list for j in w_list]
        x = torch.randn(x_cond[:, :1, :, :].shape, device=self.diffusion.device) * 0.15 # keep noise at dim 1 channel
        #x = torch.randn(x_cond[:, :1, :, :].shape, 1, self.config.data.image_size, self.config.data.image_size, device=self.diffusion.device) * 0.15
        #print(x.shape)
        #print(x_cond.shape)
        x_output = self.diffusion.sample_image(x_cond, x, latent=lt, patch_locs=corners, patch_size=p_size)
        #print(corners)
        return x_output

    def overlapping_grid_indices(self, x_cond, output_size, r=None):
        _, c, h, w = x_cond[:, :1, :, :].shape
        r = 16 if r is None else r
        h_list = [i for i in range(0, h - output_size + 1, r)]
        w_list = [i for i in range(0, w - output_size + 1, r)]
        return h_list, w_list
