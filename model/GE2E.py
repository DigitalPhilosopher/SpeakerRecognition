import torch.nn as nn
import torch.nn.functional as F

class GE2EModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, embedding_dim):
        super(GE2EModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, embedding_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        embeddings = self.fc(out[:, -1, :])
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        return normalized_embeddings
