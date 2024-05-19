import torch


def get_device(logger):
    # Check if CUDA is available
    if torch.cuda.is_available():
        logger.info("CUDA is available! Training on GPU...")
        device = torch.device("cuda")
    else:
        logger.info("CUDA is not available. Training on CPU...")
        device = torch.device("cpu")
    
    return device


def clear_gpu():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()

