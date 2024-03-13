import torch.nn as nn
import torch
import math

class AAMSoftmax(nn.Module):
    def __init__(self, in_feats, out_feats, s=30.0, m=0.4, device=None):
        super(AAMSoftmax, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.s = s
        self.m = m
        self.device = device if device else torch.device('cpu')
        self.kernel = nn.Parameter(torch.Tensor(out_feats, in_feats)).to(self.device)
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.kernel, a=math.sqrt(5))

    def forward(self, x, labels=None):
        # Normalize feature vectors and weights to lie on the hypersphere
        x_norm = torch.nn.functional.normalize(x, p=2, dim=1)
        w_norm = torch.nn.functional.normalize(self.kernel, p=2, dim=0)
        
        # Calculate cosine similarity
        cos_theta = torch.mm(x_norm, w_norm)
        
        # Additive Angular Margin
        if labels is not None:
            one_hot = torch.zeros(cos_theta.size(), device=x.device)
            one_hot.scatter_(1, labels.view(-1, 1), 1)
            cos_theta_m = cos_theta - one_hot * self.m  # Subtract margin from the true class
            output = self.s * cos_theta_m
        else:
            output = self.s * cos_theta  # Scale the logits
        
        return output