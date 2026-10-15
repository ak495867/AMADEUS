import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualUnit(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation, padding=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1)
        self.act = nn.ELU()

    def forward(self, x):
        res = x
        x = self.act(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        return x + res

class SEANetEncoder(nn.Module):
    def __init__(self, in_channels=2, base_channels=64, latent_dim=64, ratios=[8, 5, 4, 3, 2]):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.stride = 1
        for r in ratios:
            self.stride *= r

        self.initial_conv = nn.Conv1d(in_channels, base_channels, kernel_size=7, padding=3)
        layers = []
        current_ch = base_channels

        for r in ratios:
            out_ch = min(current_ch * 2, 512)
            layers.append(nn.ELU())
            layers.append(nn.Conv1d(current_ch, out_ch, kernel_size=2 * r, stride=r, padding=r // 2))
            layers.append(ResidualUnit(out_ch, dilation=1))
            layers.append(ResidualUnit(out_ch, dilation=3))
            current_ch = out_ch

        self.downsample_layers = nn.Sequential(*layers)
        self.to_mu = nn.Conv1d(current_ch, latent_dim, kernel_size=3, padding=1)
        self.to_logvar = nn.Conv1d(current_ch, latent_dim, kernel_size=3, padding=1)

    def forward(self, x, sample=True):
        h = self.initial_conv(x)
        h = self.downsample_layers(h)
        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        logvar = torch.clamp(logvar, min=-20.0, max=10.0)

        if sample:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu

        return z, mu, logvar
