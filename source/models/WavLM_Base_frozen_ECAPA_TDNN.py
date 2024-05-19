import torch
from torch import nn
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
from s3prl.hub import wavlm_base

class WavLM_Base_frozen_ECAPA_TDNN(nn.Module):
    def __init__(self, device='cuda'):
        super(WavLM_Base_frozen_ECAPA_TDNN, self).__init__()

        # Initialize WavLM Base model
        self.frontend = wavlm_base()  # Assuming s3prl's wavlm_base loader is used here
        self.frontend.to(device)

        # We need to determine the output feature size of WavLM dynamically
        # This can be done by processing a small dummy input through WavLM
        dummy_input = torch.randn(1, 16000).to(device)  # 1 second of random noise
        with torch.no_grad():
            # Forward pass to get the feature size
            dummy_output = self.frontend(dummy_input)
            # Assume we are using the last hidden state as the input to ECAPA_TDNN
            feature_size = dummy_output['hidden_states'][-1].shape[-1]

        # Initialize ECAPA_TDNN with the dynamically determined feature size
        self.embedding = ECAPA_TDNN(input_size=feature_size, lin_neurons=192, device=device)
        self.device = device

    def forward(self, batch_wavefiles):
        wavlm_features = []
        for waveform in batch_wavefiles:
            waveform = waveform.unsqueeze(0) if waveform.dim() == 1 else waveform
            with torch.no_grad():
                features = self.frontend(waveform)['hidden_states'][-1]
            wavlm_features.append(features)
        
        # Ensure all tensors are 2D before concatenation
        wavlm_features = [feat.unsqueeze(0) if feat.dim() == 1 else feat for feat in wavlm_features]
        wavlm_features = torch.cat(wavlm_features, dim=0)  # Concatenate along the batch dimension

        embeddings = self.embedding(wavlm_features)
        return embeddings