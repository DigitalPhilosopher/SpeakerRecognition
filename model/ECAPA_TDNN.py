import torch.nn as nn
import torch

class AttentiveStatPooling(nn.Module):
    def __init__(self, in_features):
        super(AttentiveStatPooling, self).__init__()
        self.in_features = in_features
        self.attention = nn.Sequential(
            nn.Conv1d(in_channels=in_features, out_channels=in_features, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(in_channels=in_features, out_channels=in_features, kernel_size=1),
            nn.Softmax(dim=2)
        )
    
    def forward(self, x):
        alpha = self.attention(x)
        mean = torch.sum(x * alpha, dim=2)
        std = torch.sqrt(torch.sum(alpha * (x - mean.unsqueeze(2))**2, dim=2))
        return torch.cat((mean, std), dim=1)
    
class ASPAndBN(nn.Module):
    def __init__(self, in_features):
        super(ASPAndBN, self).__init__()
        self.asp = AttentiveStatPooling(in_features)
        self.bn = nn.BatchNorm1d(2 * in_features)

    def forward(self, x):
        x = self.asp(x)
        x = x.unsqueeze(2)
        x = self.bn(x)
        x = x.squeeze(2)
        return x

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)
    
class SERes2Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=2, reduction=16):
        super(SERes2Block, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=dilation, dilation=dilation),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=dilation, dilation=dilation),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=dilation, dilation=dilation),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels)
        )
        self.se = SEBlock(out_channels, reduction)
        self.shortcut = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.se(out)
        return out + identity

class ECAPA_TDNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(ECAPA_TDNN, self).__init__()
        self.conv1d_relu_bn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=5, stride=1, dilation=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size)
        )
        self.se2 = SERes2Block(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, dilation=2)
        self.se22 = SERes2Block(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, dilation=3)
        self.se23 = SERes2Block(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, dilation=4)
        self.conv1d_relu = nn.Sequential(
            nn.Conv1d(in_channels=3*hidden_size, out_channels=1536, kernel_size=3, stride=1, dilation=1),
            nn.ReLU(),
        )
        self.pooling = ASPAndBN(1536)
        self.fc = nn.Sequential(
            nn.Linear(3072, 192),
            nn.BatchNorm1d(192)
        )

    def forward(self, x):
        x = self.conv1d_relu_bn(x)
        x1 = self.se2(x)
        x2 = self.se22(x1)
        x3 = self.se23(x3)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.conv1d_relu(x)
        x = self.pooling(x)
        x = self.fc(x)
        return x