#module vae_main.py
import args
import torch, numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import VaeDataset
from vae_model import ConvVAE
from utils import vae_loss_function
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
    MultiScaleStructuralSimilarityIndexMeasure
)
import matplotlib.pyplot as plt



train_latents = []
val_latents = []

num_epochs = 1#args.vae_num_epochs
batch_size=args.vae_batch_size
lr=args.vae_lr
HR=args.HR

train_data = np.load('train_data.npy')
train_targets = np.load('train_targets.npy')
val_data = np.load('val_data.npy')
val_targets = np.load('val_targets.npy')

# Use your existing Dataset class
train_dataset = VaeDataset(train_data, train_targets)
val_dataset   = VaeDataset(val_data, val_targets)
#save_dataset   = NumpyImageDataset(data, targets)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2)
train_loader_no_shuffle   = DataLoader(train_dataset,   batch_size=batch_size, shuffle=False, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if HR:
  model = ConvVAE(in_channels=1, latent_dim=64, features=[32, 64, 128], img_size=600).to(device)
else:
  model = ConvVAE(in_channels=1, latent_dim=8, features=[32, 64, 128], img_size=64).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# NEW: instantiate metrics once, move to GPU
metric_psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
metric_ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
metric_mssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device)

best_val_loss = float('inf')

for epoch in range(num_epochs):
    # -------------------- TRAIN --------------------
    model.train()
    train_loss = 0.0
    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [train]"):
        x, y = x.to(device), y.to(device)
        x, _, mean, log_var = model(x)
        loss = vae_loss_function(y, x, mean, log_var)
        #metric_mssim.reset()
        #metric_mssim.update(x, y)
        #ssim = metric_mssim.compute().item()
        metric_ssim.reset()
        metric_ssim.update(x, y)
        ssim = metric_mssim.compute().item()
        loss = loss + (1 - ssim)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * x.size(0)

    # ------------------ VALIDATION -----------------
    model.eval()
    val_loss = 0.0
    metric_psnr.reset()      # NEW
    metric_ssim.reset()      # NEW
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [val]"):
            x, y = x.to(device), y.to(device)
            x, _, mean, log_var = model(x)           # keep in [0,1]
            val_loss += vae_loss_function(y, x, mean, log_var).item() * x.size(0)
            metric_psnr.update(x, y)             # NEW
            metric_ssim.update(x, y)             # NEW

    # dataset‑wide means
    train_loss /= len(train_dataset)
    val_loss   /= len(val_dataset)
    val_psnr   = metric_psnr.compute().item()       # NEW
    val_ssim   = metric_ssim.compute().item()       # NEW

    print(f"Epoch {epoch+1}: "
          f"train_loss={train_loss:.10f} | "
          f"val_loss={val_loss:.10f} | "
          f"PSNR={val_psnr:.2f} | SSIM={val_ssim:.4f}")

    # checkpoint
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        #torch.save(model.state_dict(), "unet_best.pth")
        print("  🎉 New best validation loss, model saved!")

    # -------------- VISUALISE EVERY 3rd EPOCH --------------
    if (epoch + 1) % 10 == 0:
        n_show = 20
        inputs  = torch.stack([val_dataset[i][0] for i in range(n_show)]).to(device)
        targets = torch.stack([val_dataset[i][1] for i in range(n_show)]).to(device)
        with torch.no_grad():
            outputs,_,_,_ = model(inputs)
        # move to CPU, shape (N, H, W)
        inputs  = inputs.squeeze(1).cpu().numpy()
        outputs = outputs.squeeze(1).cpu().numpy()
        targets = targets.squeeze(1).cpu().numpy()

        fig, axes = plt.subplots(n_show, 3, figsize=(9, n_show*1.5))
        for i in range(n_show):
            axes[i, 0].imshow(inputs[i],  cmap="gray"); axes[i, 0].set_title("Noisy")
            axes[i, 1].imshow(outputs[i], cmap="gray"); axes[i, 1].set_title("Pred")
            axes[i, 2].imshow(targets[i], cmap="gray"); axes[i, 2].set_title("Clean")
            for ax in axes[i]:
                ax.axis("off")
        plt.tight_layout()
        plt.show()


for x, y in tqdm(train_loader_no_shuffle, desc=f"Generating Train Latents"):
    x, y = x.to(device), y.to(device)
    _, z, _, _ = model(x)

    # detach and move to CPU before saving
    train_latents.append(z.detach().cpu().numpy())

# stack along batch dimension
train_latents = np.concatenate(train_latents, axis=0)


for x, y in tqdm(val_loader, desc=f"Generating Validation Latents"):
    x, y = x.to(device), y.to(device)
    _, z, _, _ = model(x)

    # detach and move to CPU before saving
    val_latents.append(z.detach().cpu().numpy())

# stack along batch dimension
val_latents = np.concatenate(val_latents, axis=0)

np.save('train_latents.npy', train_latents)
np.save('val_latents.npy', val_latents)
