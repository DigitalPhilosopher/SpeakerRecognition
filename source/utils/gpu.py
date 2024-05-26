import torch


def get_device(device):
    if device:
        device = torch.device(device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return device


def list_cuda_devices():
    devices = []
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        for i in range(num_devices):
            devices.append(f"cuda:{i}")
    else:
        devices.append("cpu")

    return devices
