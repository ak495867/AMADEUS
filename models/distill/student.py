import torch
import torch.nn as nn
import torch.nn.functional as F

class DistilledDiTStudent(nn.Module):
    def __init__(self, dit_model):
        super().__init__()
        self.dit = dit_model
        self.steps = 4
        self.timesteps = [1.0, 0.75, 0.50, 0.25, 0.0]

    def compute_distillation_loss(self, teacher_model, x_noisy, t_curr, t_next, cond=None):
        dt = t_next - t_curr
        half_dt = dt * 0.5
        t_mid = t_curr + half_dt

        with torch.no_grad():
            t_curr_tensor = torch.full((x_noisy.shape[0],), t_curr, device=x_noisy.device)
            t_mid_tensor = torch.full((x_noisy.shape[0],), t_mid, device=x_noisy.device)

            v1 = teacher_model(x_noisy, t_curr_tensor, **(cond if cond is not None else {}))
            x_mid = x_noisy + half_dt * v1
            v2 = teacher_model(x_mid, t_mid_tensor, **(cond if cond is not None else {}))
            x_target = x_noisy + dt * (0.5 * (v1 + v2))

        v_student = self.dit(x_noisy, t_curr_tensor, **(cond if cond is not None else {}))
        x_student = x_noisy + dt * v_student

        return F.mse_loss(x_student, x_target)

    @torch.no_grad()
    def sample(self, shape, device, cond=None, cfg_scale=1.5):
        x = torch.randn(shape, device=device)
        for i in range(self.steps):
            t_curr = self.timesteps[i]
            t_next = self.timesteps[i + 1]
            dt = t_next - t_curr

            t_tensor = torch.full((shape[0],), t_curr, device=device)

            if cond is not None:
                v_cond = self.dit(x, t_tensor, **cond)
                uncond = {k: None for k in cond}
                v_uncond = self.dit(x, t_tensor, **uncond)
                v_pred = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v_pred = self.dit(x, t_tensor)

            x = x + dt * v_pred

        return x
