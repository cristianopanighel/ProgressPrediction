import torch

def load_splitfile(path: str):
    with open(path) as f:
        names = f.readlines()
    return [name.strip() for name in names]

def get_device() -> torch.device:
    # if torch.backends.mps.is_available():
    #    device = "mps"
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    return torch.device(device)
