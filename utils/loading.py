import torch
import numpy as np


def save_model_npz(model, path="compressed_models/compressed.npz"):
    """
    Saves model weights + masks to .npz format.
    Smaller than .pth because pruned weights are stored sparsely.
    """
    arrays = {}
    for name, param in model.named_parameters():
        arrays[name] = param.data.cpu().numpy()

    # Also save masks from pruned layers
    for name, buf in model.named_buffers():
        if "mask" in name:
            arrays[f"mask_{name}"] = buf.cpu().numpy()

    np.savez_compressed(path, **arrays)
    print(f"[Loading] Model saved to {path}")


def load_model_from_npz(model, path, device):
    """
    Loads weights + masks back into model from .npz file.
    """
    data      = np.load(path)
    state     = model.state_dict()
    buf_state = {name: buf for name, buf in model.named_buffers()}

    for name, param in model.named_parameters():
        if name in data:
            param.data = torch.tensor(
                data[name], dtype=param.dtype
            ).to(device)

    # Restore masks
    for name, buf in model.named_buffers():
        key = f"mask_{name}"
        if key in data:
            buf.copy_(torch.tensor(data[key], dtype=buf.dtype).to(device))

    model.to(device)
    print(f"[Loading] Model loaded from {path}")
    return model