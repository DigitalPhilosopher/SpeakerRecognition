from torch import nn
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN

class SpeakerClassifier(nn.Module):
    def __init__(self, num_speakers, input_size=80, embedding_size=192, device='cuda'):
        super(SpeakerClassifier, self).__init__()

        self.embedding = ECAPA_TDNN(input_size=input_size, lin_neurons=embedding_size, device=device)
        self.classifier = nn.Linear(embedding_size, num_speakers)

    def forward(self, x):
        x = self.embedding(x)
        x = self.classifier(x)
        return x