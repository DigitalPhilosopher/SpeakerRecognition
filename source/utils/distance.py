import torch


def l2_normalize(tensor):
    norm = torch.norm(tensor, p=2, dim=-1, keepdim=True)
    normalized_tensor = tensor / norm
    return normalized_tensor


def compute_distance(emb1, emb2):
    emb1 = torch.squeeze(emb1)
    emb2 = torch.squeeze(emb2)
    return torch.sum((emb1 - emb2) ** 2)
