import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class FlowMatchingScheduler:
    def __init__(self, sigma_min=1e-4, logit_normal_mean=0.0, logit_normal_std=1.0):
        self.sigma_min = sigma_min
        self.logit_normal_mean = logit_normal_mean
        self.logit_normal_std = logit_normal_std

    def sample_timesteps(self, batch_size, device):
        u = torch.randn(batch_size, device=device) * self.logit_normal_std + self.logit_normal_mean
        t = torch.sigmoid(u)
        return t

    def add_noise(self, x_0, noise, t):
        t_expand = t.view(-1, 1, 1)
        x_t = (1.0 - (1.0 - self.sigma_min) * t_expand) * x_0 + t_expand * noise
        target_v = noise - (1.0 - self.sigma_min) * x_0
        return x_t, target_v

    @torch.no_grad()
    def sample_euler(
        self,
        model,
        shape,
        device,
        steps=32,
        cond=None,
        cfg_scale=3.5,
        melody_cfg_scale=2.0
    ):
        x = torch.randn(shape, device=device)
        timesteps = torch.linspace(1.0, 0.0, steps + 1, device=device)

        for i in range(steps):
            t_curr = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t_curr

            t_tensor = torch.full((shape[0],), t_curr, device=device)

            if cond is not None:
                v_cond = model(x, t_tensor, **cond)

                uncond = {k: None for k in cond}
                v_uncond = model(x, t_tensor, **uncond)

                if "melody" in cond and cond["melody"] is not None:
                    no_mel_cond = {k: v for k, v in cond.items()}
                    no_mel_cond["melody"] = None
                    v_no_mel = model(x, t_tensor, **no_mel_cond)
                    v_pred = v_uncond + cfg_scale * (v_no_mel - v_uncond) + melody_cfg_scale * (v_cond - v_no_mel)
                else:
                    v_pred = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v_pred = model(x, t_tensor)

            x = x + dt * v_pred

        return x
