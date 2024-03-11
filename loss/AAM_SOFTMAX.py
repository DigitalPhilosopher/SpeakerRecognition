import torch.nn as nn
import torch
import math

class AAMSoftmax(nn.Module):
    def __init__(self, in_feats, out_feats, s=30.0, m=0.4):
        super(AAMSoftmax, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.s = s
        self.m = m
        self.kernel = nn.Parameter(torch.Tensor(in_feats, out_feats))
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.kernel, a=math.sqrt(5))