import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AAMSoftmax(nn.Module):
    def __init__(self, in_feats, out_feats, s=30.0, m=0.4, device=None):
        super(AAMSoftmax, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.s = s  # scale factor
        self.m = m  # margin
        self.device = device if device else torch.device('cpu')
        self.kernel = nn.Parameter(torch.Tensor(out_feats, in_feats)).to(self.device)
        nn.init.kaiming_uniform_(self.kernel, a=math.sqrt(5))


    def forward(self, x, labels):
        # Normalize feature vectors and weights to lie on the hypersphere
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.kernel, p=2, dim=1)
        
        # Calculate cosine similarity
        cos_theta = torch.mm(x_norm, w_norm.t())
        
        # Additive Angular Margin
        one_hot = torch.zeros(cos_theta.size(), device=self.device)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        with torch.no_grad():
            original_target_logit = cos_theta[torch.arange(0, x.size(0)), labels].view(-1, 1)
        cos_theta_m = cos_theta - one_hot * self.m  # Subtract margin from the true class

        # Apply scale
        cos_theta_m_scaled = self.s * cos_theta_m

        # Cross-Entropy Loss
        loss = F.cross_entropy(cos_theta_m_scaled, labels)
        
        return loss
