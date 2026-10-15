import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=[1024, 2048, 512], hop_sizes=[120, 240, 50], win_lengths=[600, 1200, 240]):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths

    def stft(self, x, fft_size, hop_size, win_length):
        if x.dim() == 3:
            b, c, t = x.shape
            x = x.reshape(b * c, t)
        window = torch.hann_window(win_length, device=x.device)
        s = torch.stft(x, n_fft=fft_size, hop_length=hop_size, win_length=win_length, window=window, return_complex=True)
        mag = torch.abs(s)
        return mag

    def forward(self, x_pred, x_true):
        min_len = min(x_pred.shape[-1], x_true.shape[-1])
        x_pred = x_pred[..., :min_len]
        x_true = x_true[..., :min_len]

        loss_sc = 0.0
        loss_mag = 0.0
        for n_fft, hop, win in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            mag_pred = self.stft(x_pred, n_fft, hop, win)
            mag_true = self.stft(x_true, n_fft, hop, win)

            sc = torch.norm(mag_true - mag_pred, p="fro") / (torch.norm(mag_true, p="fro") + 1e-7)
            mag_l1 = F.l1_loss(torch.log(mag_pred + 1e-7), torch.log(mag_true + 1e-7))

            loss_sc = loss_sc + sc
            loss_mag = loss_mag + mag_l1

        return (loss_sc + loss_mag) / len(self.fft_sizes)

class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods=[2, 3, 5, 7, 11]):
        super().__init__()
        self.periods = periods
        self.discriminators = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 32, (5, 1), (3, 1), padding=(2, 0)),
                nn.LeakyReLU(0.2),
                nn.Conv2d(32, 128, (5, 1), (3, 1), padding=(2, 0)),
                nn.LeakyReLU(0.2),
                nn.Conv2d(128, 512, (5, 1), (3, 1), padding=(2, 0)),
                nn.LeakyReLU(0.2),
                nn.Conv2d(512, 1024, (5, 1), (1, 1), padding=(2, 0)),
                nn.LeakyReLU(0.2),
                nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)),
            )
            for _ in periods
        ])

    def forward(self, x):
        features = []
        scores = []
        if x.dim() == 3:
            x = x.mean(dim=1, keepdim=True)
        for p, disc in zip(self.periods, self.discriminators):
            b, c, t = x.shape
            pad = (p - (t % p)) % p
            x_pad = F.pad(x, (0, pad))
            x_reshaped = x_pad.view(b, c, -1, p)
            f_list = []
            h = x_reshaped
            for layer in disc:
                h = layer(h)
                f_list.append(h)
            scores.append(h)
            features.append(f_list)
        return scores, features

class CodecLossSuite(nn.Module):
    def __init__(self, kl_weight=1e-4, stft_weight=45.0, l1_weight=1.0, adv_weight=1.0):
        super().__init__()
        self.kl_weight = kl_weight
        self.stft_weight = stft_weight
        self.l1_weight = l1_weight
        self.adv_weight = adv_weight
        self.stft_loss = MultiResolutionSTFTLoss()
        self.mpd = MultiPeriodDiscriminator()

    def kl_divergence(self, mu, logvar):
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1, 2])
        return kl.mean()

    def generator_loss(self, x_pred, x_true, mu, logvar):
        min_len = min(x_pred.shape[-1], x_true.shape[-1])
        x_pred = x_pred[..., :min_len]
        x_true = x_true[..., :min_len]

        loss_l1 = F.l1_loss(x_pred, x_true)
        loss_stft = self.stft_loss(x_pred, x_true)
        loss_kl = self.kl_divergence(mu, logvar)

        scores_fake, feat_fake = self.mpd(x_pred)
        _, feat_real = self.mpd(x_true)

        loss_adv = 0.0
        for s in scores_fake:
            loss_adv = loss_adv + torch.mean((1.0 - s) ** 2)

        loss_fm = 0.0
        for f_f, f_r in zip(feat_fake, feat_real):
            for l_f, l_r in zip(f_f, f_r):
                loss_fm = loss_fm + F.l1_loss(l_f, l_r.detach())

        total = (self.l1_weight * loss_l1 +
                 self.stft_weight * loss_stft +
                 self.kl_weight * loss_kl +
                 self.adv_weight * loss_adv +
                 2.0 * loss_fm)

        return total, {
            "loss_total": total.item(),
            "loss_l1": loss_l1.item(),
            "loss_stft": loss_stft.item(),
            "loss_kl": loss_kl.item(),
            "loss_adv": loss_adv.item(),
            "loss_fm": loss_fm.item()
        }

    def discriminator_loss(self, x_pred, x_true):
        min_len = min(x_pred.shape[-1], x_true.shape[-1])
        x_pred = x_pred[..., :min_len]
        x_true = x_true[..., :min_len]

        scores_fake, _ = self.mpd(x_pred.detach())
        scores_real, _ = self.mpd(x_true)

        loss_d = 0.0
        for s_f, s_r in zip(scores_fake, scores_real):
            loss_d = loss_d + torch.mean(s_f ** 2) + torch.mean((1.0 - s_r) ** 2)
        return loss_d
