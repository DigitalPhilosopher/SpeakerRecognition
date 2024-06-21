import torch
import torch.nn.functional as F

def l2_normalize(tensor):
    norm = torch.norm(tensor, p=2, dim=-1, keepdim=True)
    normalized_tensor = tensor / norm
    return normalized_tensor


def compute_distance(emb1, emb2):
    # Reshape if necessary
    if emb1.dim() == 3:
        emb1 = emb1.view(emb1.size(0), -1)
        emb2 = emb2.view(emb2.size(0), -1)
    dist = torch.sum((emb1 - emb2) ** 2, -1)
    return dist



# def compute_distance(emb1, emb2):
#     emb1 = torch.squeeze(emb1)
#     emb2 = torch.squeeze(emb2)
#     return torch.sum((emb1 - emb2) ** 2)

