import torch


def get_device(device):
    if device:
        device = torch.device(device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return device


def clear_gpu():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
