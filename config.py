import torch

def config_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Config] Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"[Config] VRAM: "
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[Config] Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("[Config] Using CPU")
    return device