#module train_diffusion.py

import os
import args
import yaml
import torch
import numpy as np
import torchvision
#import diffusion_unet
import dataset
#import utils
from denoising_diffusion import DenoisingDiffusion

# ✅ Directly specify these values instead of using argparse
CONFIG_PATH = args.CONFIG_PATH                   # Your YAML config file
RESUME_PATH = args.RESUME_PATH                   # Path to checkpoint (if resuming)
SAMPLING_TIMESTEPS = args.SAMPLING_TIMESTEPS     # Validation sampling steps
IMAGE_FOLDER = args.IMAGE_FOLDER                 # Where to save validation images
SEED = args.SEED                                 # Random seed


# ✅ Define Args class at the top-level
class Args:
    def __init__(self, resume, sampling_timesteps, image_folder, seed):
        self.resume = resume
        self.sampling_timesteps = sampling_timesteps
        self.image_folder = image_folder
        self.seed = seed


def dict2namespace(config):
    import argparse
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def load_config_and_args():
    # Load YAML config
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    config_ns = dict2namespace(config)

    # Return Args instance
    args = Args(
        resume=RESUME_PATH,
        sampling_timesteps=SAMPLING_TIMESTEPS,
        image_folder=IMAGE_FOLDER,
        seed=SEED
    )
    return args, config_ns


def main():
    args, config = load_config_and_args()

    train_data = np.load('train_data.npy')
    train_latents = np.load('train_latents.npy')
    train_targets = np.load('train_targets.npy')
    
    val_data = np.load('val_data.npy')
    val_latents = np.load('val_latents.npy')
    val_targets = np.load('val_targets.npy')

    # setup device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device:", device)
    config.device = device

    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    # load dataset
    print("=> using dataset '{}'".format(config.data.dataset))
    DATASET = dataset.Astro(config, train_data, train_latents, train_targets, val_data, val_latents, val_targets)

    # create model
    print("=> creating denoising-diffusion model...")
    diffusion = DenoisingDiffusion(args, config)
    diffusion.train(DATASET)


if __name__ == "__main__":
    main()
