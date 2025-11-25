#@title Dataset Class

import os
from os import listdir
from os.path import isfile
import torch
import numpy as np
import torchvision
import torch.utils.data
import PIL
import re
import random

class VaeDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets):
        assert inputs.shape == targets.shape
        self.inputs = inputs.astype(np.float32)
        self.targets = targets.astype(np.float32)

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        x = self.inputs[idx][None, ...]  # shape 1×600×600
        y = self.targets[idx][None, ...]
        return torch.tensor(x), torch.tensor(y)

class AstroDataset(torch.utils.data.Dataset):
    def __init__(self, input_array, latent_array, gt_array, patch_size, n, transforms, parse_patches=True):
        super().__init__()
        assert input_array.shape == gt_array.shape
        self.input_array = input_array  # Shape: (N, H, W)
        self.latent_array = latent_array
        self.gt_array = gt_array
        self.patch_size = patch_size
        self.transforms = transforms
        self.n = n
        self.parse_patches = parse_patches

    @staticmethod
    def get_params(h, w, output_size, n):
        th, tw = output_size
        i_list = [random.randint(0, h - th) for _ in range(n)]
        j_list = [random.randint(0, w - tw) for _ in range(n)]
        return i_list, j_list, th, tw

    @staticmethod
    def n_random_crops(img, i_list, j_list, h, w):
        crops = [img[i:i+h, j:j+w] for i, j in zip(i_list, j_list)]
        return crops

    def get_images(self, index):
        input_img = self.input_array[index]  # Shape: (H, W)
        gt_img = self.gt_array[index]        # Shape: (H, W)
        lt_img = self.latent_array[index]       # Shape: (H)
        
        lt_img = np.tile(lt_img[:, None], (1, lt_img.shape[0]))
        lt_img = self.transforms(lt_img[np.newaxis, ...])
        img_id = f"{index:05d}"

        if self.parse_patches:
            h, w = input_img.shape
            i, j, ph, pw = self.get_params(h, w, (self.patch_size, self.patch_size), self.n)
            input_crops = self.n_random_crops(input_img, i, j, ph, pw)
            gt_crops = self.n_random_crops(gt_img, i, j, ph, pw)

            outputs = []
            for inp, gt in zip(input_crops, gt_crops):
                inp_tensor = self.transforms(inp[np.newaxis, ...])  # Add channel dim
                gt_tensor = self.transforms(gt[np.newaxis, ...])
                outputs.append(torch.cat([inp_tensor, lt_img, gt_tensor], dim=0))
            return torch.stack(outputs, dim=0), img_id
        else:
            # Resize to multiples of 16
            h, w = input_img.shape
            if h > w and h > 1024:
                w = int(np.ceil(w * 1024 / h))
                h = 1024
            elif w >= h and w > 1024:
                h = int(np.ceil(h * 1024 / w))
                w = 1024
            h = int(16 * np.ceil(h / 16.0))
            w = int(16 * np.ceil(w / 16.0))

            input_resized = torch.nn.functional.interpolate(
                torch.from_numpy(input_img[np.newaxis, np.newaxis, ...]).float(), size=(h, w), mode='bilinear', align_corners=False
            ).squeeze(0)

            gt_resized = torch.nn.functional.interpolate(
                torch.from_numpy(gt_img[np.newaxis, np.newaxis, ...]).float(), size=(h, w), mode='bilinear', align_corners=False
            ).squeeze(0)

            return torch.cat([input_resized, gt_resized], dim=0), lt_img, img_id

    def __getitem__(self, index):
        return self.get_images(index)

    def __len__(self):
        return self.input_array.shape[0]


class Astro:
    def __init__(self, config, train_input_array, train_lt_array, train_gt_array, val_input_array, val_lt_array, val_gt_array):
        self.config = config
        self.train_input_array = train_input_array
        self.train_lt_array = train_lt_array
        self.train_gt_array = train_gt_array
        self.val_input_array = val_input_array
        self.val_lt_array = val_lt_array
        self.val_gt_array = val_gt_array
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda x: torch.from_numpy(x).float())
        ])

    def get_loaders(self, parse_patches=True):
        print("=> loading grayscale numpy Astro dataset...")

        train_dataset = AstroDataset(
            input_array=self.train_input_array,
            latent_array = self.train_lt_array,
            gt_array=self.train_gt_array,
            n=self.config.training.patch_n,
            patch_size=self.config.data.image_size,
            transforms=self.transforms,
            parse_patches=parse_patches
        )

        val_dataset = AstroDataset(
            input_array=self.val_input_array,
            latent_array = self.val_lt_array,
            gt_array=self.val_gt_array,
            n=self.config.training.patch_n,
            patch_size=self.config.data.image_size,
            transforms=self.transforms,
            parse_patches=parse_patches
        )

        if not parse_patches:
            self.config.training.batch_size = 128
            self.config.sampling.batch_size = 128

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                   shuffle=True, num_workers=self.config.data.num_workers,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.sampling.batch_size,
                                                 shuffle=False, num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return train_loader, val_loader
