import os
import gdown
import args
import numpy as np
from sklearn.model_selection import train_test_split

HR= args.HR

if HR:
    targ_url = args.targ_url_hr
    data_url = args.data_url_hr
    output_data = 'data.npy'
    output_targ = 'target.npy'
    gdown.download(url=data_url, output=output_data, fuzzy=True)
    gdown.download(url=targ_url, output=output_targ, fuzzy=True)
    data = np.load("data.npy")
    targets = np.load("target.npy")
else:
    targ_url = args.targ_url_lr
    data_url = args.data_url_lr
    output_data = 'resized_data.npy'
    output_targ = 'resized_target.npy'
    gdown.download(url=data_url, output=output_data, fuzzy=True)
    gdown.download(url=targ_url, output=output_targ, fuzzy=True)
    data = np.load("resized_data.npy")
    targets = np.load("resized_target.npy")


# Split keeping correspondences
train_data, val_data, train_targets, val_targets = train_test_split(
    data, targets,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

np.save('train_data.npy', train_data)
np.save('val_data.npy', val_data)
np.save('train_targets.npy', train_targets)
np.save('val_targets.npy', val_targets)
