#@title Hyperparameters

import args
import yaml

config = {
    "data": {
        "dataset": "Astro",
        "image_size": 64 if args.HR else 8,
        "channels": 1,
        "num_workers": 1,
        "data_dir": "/content/",
        "conditional": True,
        "global_conditional": True
    },
    "model": {
        "in_channels": 1,
        "out_ch": 1,
        "ch": 128,
        "ch_mult": [1, 1, 2, 2, 4, 4] if args.HR else [1, 2, 2, 4],
        "num_res_blocks": 2,
        "attn_resolutions": [16],
        "dropout": 0.0,
        "ema_rate": 0.999,
        "ema": False,
        "resamp_with_conv": True
    },
    "diffusion": {
        "beta_schedule": "linear",
        "beta_start": 0.0001,
        "beta_end": 0.02,
        "num_diffusion_timesteps": 1000
    },
    "training": {
        "patch_n": 8,
        "batch_size": 8,
        "n_epochs": 4000,
        "n_iters": 2000000,
        "snapshot_freq": 40000,
        "validation_freq": 40000
    },
    "sampling": {
        "batch_size": 4,
        "last_only": True
    },
    "optim": {
        "weight_decay": 0.0,
        "optimizer": "Adam",
        "lr": 0.00002,
        "amsgrad": False,
        "eps": 1e-8
    }
}

with open("config.yml", "w") as file:
    yaml.dump(config, file, default_flow_style=False)

print("config.yml created.")

class DotDict(dict):
    """Allows dot notation access to dictionary attributes."""
    def __getattr__(self, attr):
        value = self.get(attr)
        if isinstance(value, dict):
            return DotDict(value)
        return value

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

config = DotDict(config)
