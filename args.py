# args
HR = False

targ_url_hr = 'https://drive.google.com/file/d/1g3Q0atGa_Leyt5udIFXs5a7HsCL-gGmv/view?usp=sharing'
data_url_hr = 'https://drive.google.com/file/d/1EYZzQg-PNXWN1vpD0BLSbfG6jyP0nTVr/view?usp=sharing'

targ_url_lr = 'https://drive.google.com/file/d/1l8zYsfQ9vXj5AxGeQFy1eHrbrHRgVKkY/view?usp=sharing'
data_url_lr = 'https://drive.google.com/file/d/14BZ9PBw7zUaoS2Uy4WXIaKwDRN0A3Dn3/view?usp=sharing'

vae_num_epochs = 20
vae_batch_size= 8
vae_lr=1e-4

# Diffusion model arguments
CONFIG_PATH = "config.yml"         # Your YAML config file

# training argments
RESUME_PATH = ""                   # Path to checkpoint (if resuming)
SAMPLING_TIMESTEPS = 1000            # Validation sampling steps
IMAGE_FOLDER = "results/images/"   # Where to save validation images
SEED = 61                          # Random seed

# validation arguments
val_resume_path = '/content/ckpts/Astro_ddpm.pth.tar'  # Path to model checkpoint if needed
grid_r = 16 if HR else 4
sampling_timesteps = 50
train_set = 'train_set_reconstruction'
test_set = 'validation_set_reconstruction'
image_folder = 'results/images/'
# --------------------------------

