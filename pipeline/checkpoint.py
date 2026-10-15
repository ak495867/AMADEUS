import os
import torch

def save_model_checkpoint(model, filepath, optimizer=None, epoch=0, step=0, extra_meta=None):
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    state = {
        "model_state": model.state_dict(),
        "epoch": epoch,
        "step": step,
    }
    if optimizer is not None:
        state["optimizer_state"] = optimizer.state_dict()
    if extra_meta is not None:
        state["extra_meta"] = extra_meta
    torch.save(state, filepath)

def load_model_checkpoint(model, filepath, optimizer=None, device="cpu"):
    if not os.path.exists(filepath):
        return None
    state = torch.load(filepath, map_location=device)
    raw_dict = state["model_state"] if "model_state" in state else state
    model_dict = model.state_dict()
    filtered_dict = {}
    for k, v in raw_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            filtered_dict[k] = v
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)
    if optimizer is not None and "optimizer_state" in state:
        try:
            optimizer.load_state_dict(state["optimizer_state"])
        except Exception:
            pass
    return state
