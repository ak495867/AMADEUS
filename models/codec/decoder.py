import torch
import torch.nn as nn
import torch.nn.functional as F

class MRFBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilations=[1, 3, 5]):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.Conv1d(channels, channels, kernel_size=kernel_size, dilation=d, padding=(kernel_size - 1) * d // 2),
                nn.LeakyReLU(0.2),
                nn.Conv1d(channels, channels, kernel_size=1)
            )
            for d in dilations
        ])

    def forward(self, x):
        out = 0
        for conv in self.convs:
            out = out + conv(x)
        return out

class HiFiGANCodecDecoder(nn.Module):
    def __init__(self, latent_dim=64, base_channels=512, out_channels=2, ratios=[2, 3, 4, 5, 8], enable_stems=True):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.enable_stems = enable_stems
        self.stem_names = ["drums", "bass", "harmony", "lead"]

        self.initial_conv = nn.Conv1d(latent_dim, base_channels, kernel_size=7, padding=3)

        self.upsample_blocks = nn.ModuleList()
        self.mrf_blocks = nn.ModuleList()

        current_ch = base_channels
        for r in ratios:
            out_ch = current_ch // 2
            self.upsample_blocks.append(
                nn.Sequential(
                    nn.LeakyReLU(0.2),
                    nn.ConvTranspose1d(current_ch, out_ch, kernel_size=2 * r, stride=r, padding=r // 2)
                )
            )
            self.mrf_blocks.append(
                nn.ModuleList([
                    MRFBlock(out_ch, kernel_size=3),
                    MRFBlock(out_ch, kernel_size=7),
                    MRFBlock(out_ch, kernel_size=11),
                ])
            )
            current_ch = out_ch

        self.final_act = nn.LeakyReLU(0.2)
        self.final_conv = nn.Conv1d(current_ch, out_channels, kernel_size=7, padding=3)

        if enable_stems:
            self.stem_heads = nn.ModuleDict({
                stem: nn.Sequential(
                    nn.LeakyReLU(0.2),
                    nn.Conv1d(current_ch, current_ch // 2, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2),
                    nn.Conv1d(current_ch // 2, out_channels, kernel_size=7, padding=3),
                    nn.Tanh()
                )
                for stem in self.stem_names
            })

    def decode_latents(self, z):
        h = self.initial_conv(z)
        for up, mrfs in zip(self.upsample_blocks, self.mrf_blocks):
            h = up(h)
            mrf_sum = 0
            for mrf in mrfs:
                mrf_sum = mrf_sum + mrf(h)
            h = mrf_sum / len(mrfs)
        return h

    def forward(self, z, return_stems=False):
        features = self.decode_latents(z)
        main_audio = torch.tanh(self.final_conv(self.final_act(features)))

        if return_stems and self.enable_stems:
            stems = {}
            for name, head in self.stem_heads.items():
                stems[name] = head(features)
            return main_audio, stems

        return main_audio
