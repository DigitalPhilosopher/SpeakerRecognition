import torch 
import torch.nn as nn
import torch.nn.functional as F

class GE2ELoss(nn.Module):
    def __init__(self, init_w=10.0, init_b=-5.0):
        super(GE2ELoss, self).__init__()
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))

    def forward(self, embeddings, num_speakers, num_utterances):
        assert embeddings.size(0) == num_speakers * num_utterances
        embeddings = embeddings.view(num_speakers, num_utterances, -1)
        
        centroids = torch.mean(embeddings, 1)
        centroids_exp = centroids.unsqueeze(1).expand(-1, num_utterances, -1)
        embeddings_exp = embeddings.unsqueeze(0).expand(num_speakers, -1, -1, -1)
        
        sim_matrix = self.w * F.cosine_similarity(embeddings_exp, centroids_exp, dim=-1) + self.b
        
        mask = 1 - torch.eye(num_speakers).expand(num_speakers, num_utterances, num_speakers).to(embeddings.device)
        sim_matrix = sim_matrix * mask
        
        correct_scores = torch.diagonal(sim_matrix, 0, 1, 2).unsqueeze(-1)
        wrong_scores = sim_matrix.masked_fill_(mask == 0, float('-inf'))
        
        loss = F.softplus(torch.logsumexp(wrong_scores, dim=-1) - correct_scores).mean()
        return loss