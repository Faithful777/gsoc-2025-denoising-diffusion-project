#module validate.py

import os
import args
import random
import socket
import yaml
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
#import models
import datasets
#import utils
import dataset
from types import SimpleNamespace
from train_diffusion import Args
from denoising_diffusion import DenoisingDiffusion
from restoration import DiffusiveRestoration


def dict2namespace(config):
    namespace = SimpleNamespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    # ----- ARGUMENTS -----
    # Store values from the imported args module before creating a local 'args' variable
    # The local 'args_ns' will then be used for the rest of the main function
    args_ns = SimpleNamespace(
        config=args.CONFIG_PATH,
        resume=args.val_resume_path,
        grid_r=args.grid_r,
        sampling_timesteps=args.sampling_timesteps,
        train_set=args.train_set,
        test_set=args.test_set,
        image_folder=args.image_folder,
        seed=args.SEED
    )
    # --------------------------------

    # load config
    with open(args_ns.config, "r") as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)

    # setup device to run
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))
    config.device = device

    if torch.cuda.is_available():
        print('Note: Currently supports evaluations (restoration) when run only on a single GPU!')

    # set random seed
    torch.manual_seed(args_ns.seed)
    np.random.seed(args_ns.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_ns.seed)
    torch.backends.cudnn.benchmark = True

    # data loading
    print("=> using dataset '{}'".format(config.data.dataset))
    DATASET = dataset.Astro(config, train_data, train_latents, train_targets, val_data, val_latents, val_targets)
    train_loader, val_loader = DATASET.get_loaders(parse_patches=False)

    # create model
    print("=> creating denoising-diffusion model with wrapper...")
    diffusion = DenoisingDiffusion(args_ns, config)
    model = DiffusiveRestoration(diffusion, args_ns, config)
    model.restore(train_loader, validation=args_ns.train_set, r=args_ns.grid_r)
    model.restore(val_loader, validation=args_ns.test_set, r=args_ns.grid_r,show_images=True)


if __name__ == '__main__':
    train_data = np.load('train_data.npy')
    train_latents = np.load('train_latents.npy')
    train_targets = np.load('train_targets.npy')

    val_data = np.load('val_data.npy')
    val_latents = np.load('val_latents.npy')
    val_targets = np.load('val_targets.npy')

    main()
