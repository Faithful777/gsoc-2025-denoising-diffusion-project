import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class ConvVAE(nn.Module):
    def __init__(self, in_channels=1, latent_dim=128, features=[32, 64, 128], img_size=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.features = features
        self.img_size = img_size
        self.encoder_convs = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)

        curr_channels = in_channels
        for feat in features:
            self.encoder_convs.append(DoubleConv(curr_channels, feat))
            curr_channels = feat

        self.bottleneck_conv = DoubleConv(features[-1], features[-1] * 2)
        self.flatten = nn.Flatten()

        conv_output_size = img_size // (2 ** len(features))  # assuming max pooling halves each time
        bottleneck_dim = (features[-1] * 2) * conv_output_size * conv_output_size

        # Latent representation
        self.fc_mu = nn.Linear(bottleneck_dim, latent_dim)
        self.fc_logvar = nn.Linear(bottleneck_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, bottleneck_dim)

        # Decoder
        self.up_convs = nn.ModuleList()
        self.up_transpose = nn.ModuleList()
        rev_feats = features[::-1]
        curr_channels = features[-1] * 2

        for feat in rev_feats:
            self.up_transpose.append(nn.ConvTranspose2d(curr_channels, feat, kernel_size=2, stride=2))
            self.up_convs.append(DoubleConv(feat * 2, feat))
            curr_channels = feat

        self.final_conv = nn.Conv2d(features[0], in_channels, kernel_size=1)

    def encode(self, x):
        skips = []
        for conv in self.encoder_convs:
            x = conv(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck_conv(x)
        x_flat = self.flatten(x)
        mu = self.fc_mu(x_flat)
        logvar = self.fc_logvar(x_flat)
        return mu, logvar, skips, x.shape[2:]  # also return spatial size

    def reparameterize(self, mu, logvar):
        if self.training:
          std = torch.exp(0.5 * logvar)
          eps = torch.randn_like(std)
          return mu + eps * std
        else:
          return mu

    def decode(self, z, skips, bottleneck_shape):
        x = self.fc_decode(z)
        x = x.view(z.size(0), self.features[-1]*2, *bottleneck_shape)
        for up, conv, skip in zip(self.up_transpose, self.up_convs, reversed(skips)):
            x = up(x)
            if x.shape[2:] != skip.shape[2:]:  # fix size mismatch (padding issue)
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = conv(x)
        return torch.sigmoid(self.final_conv(x))

    def forward(self, x):
        mu, logvar, skips, shape = self.encode(x)
        z = self.reparameterize(mu, logvar)
        #print(z.shape)
        x_hat = self.decode(z, skips, shape)
        return x_hat, z, mu, logvar
