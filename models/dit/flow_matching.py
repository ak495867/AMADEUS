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
        melody_cfg_scale=2.0,
        cfg_rescale=0.7
    ):
        x = torch.randn(shape, device=device)
        timesteps = torch.linspace(1.0, 0.0, steps + 1, device=device)

        for i in range(steps):
            t_curr = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t_curr

            t_tensor = torch.full((shape[0],), t_curr, device=device)

            if cond is not None and cfg_scale > 1.0:
                x_in = torch.cat([x, x], dim=0)
                t_in = torch.cat([t_tensor, t_tensor], dim=0)

                batched_cond = {}
                for k, v in cond.items():
                    if v is not None and isinstance(v, torch.Tensor):
                        batched_cond[k] = torch.cat([v, torch.zeros_like(v)], dim=0)
                    else:
                        batched_cond[k] = None

                v_out = model(x_in, t_in, **batched_cond)
                v_cond, v_uncond = v_out.chunk(2, dim=0)

                v_pred = v_uncond + cfg_scale * (v_cond - v_uncond)

                if cfg_rescale > 0.0:
                    std_cond = torch.std(v_cond, dim=(-1, -2), keepdim=True)
                    std_pred = torch.std(v_pred, dim=(-1, -2), keepdim=True)
                    rescaled = v_pred * (std_cond / (std_pred + 1e-7))
                    v_pred = (1.0 - cfg_rescale) * v_pred + cfg_rescale * rescaled
            else:
                v_pred = model(x, t_tensor, **(cond if cond is not None else {}))

            x = x + dt * v_pred

        return x
