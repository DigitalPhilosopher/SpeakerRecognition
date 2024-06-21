import torch
from torch import nn
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
from abc import ABC, abstractmethod


class s3prl_ECAPA_TDNN(nn.Module, ABC):
    def __init__(self, frozen=True, device='cuda'):
        super().__init__()

        # Initialize WavLM Base model
        self.frontend = self.hub_function()
        self.frontend.to(device)

        # Freeze the frontend parameters
        self.frozen = frozen
        if frozen:
            for param in self.frontend.parameters():
                param.requires_grad = False

        # We need to determine the output feature size of WavLM dynamically
        # This can be done by processing a small dummy input through WavLM
        dummy_input = torch.randn(1, 16000).to(
            device)  # 1 second of random noise
        with torch.no_grad():
            # Forward pass to get the feature size
            # dummy_output = self.frontend(dummy_input)
            dummy_output = self.extract_feat_S3PRL(dummy_input)
            # Assume we are using the last hidden state as the input to ECAPA_TDNN
            # feature_size = dummy_output['hidden_states'][-1].shape[-1]
            feature_size = dummy_output.shape[-1]

        # Initialize ECAPA_TDNN with the dynamically determined feature size
        self.embedding = ECAPA_TDNN(
            input_size=feature_size, lin_neurons=192, device=device)
        self.device = device
    #
    # def forward(self, batch_wavefiles):
    #     wavlm_features = []
    #     for waveform in batch_wavefiles:
    #         waveform = waveform.unsqueeze(
    #             0) if waveform.dim() == 1 else waveform
    #         if self.frozen:
    #             with torch.no_grad():
    #                 # features = self.frontend(waveform)['hidden_states'][-1]
    #                 features = self.extract_feat_S3PRL(waveform)['hidden_states'][-1]
    #         else:
    #             # features = self.frontend(waveform)['hidden_states'][-1]
    #             features = self.extract_feat_S3PRL(waveform)['hidden_states'][-1]
    #         wavlm_features.append(features)
    #
    #     # Ensure all tensors are 2D before concatenation
    #     wavlm_features = [feat.unsqueeze(0) if feat.dim(
    #     ) == 1 else feat for feat in wavlm_features]
    #     # Concatenate along the batch dimension
    #     wavlm_features = torch.cat(wavlm_features, dim=0)
    #
    #     embeddings = self.embedding(wavlm_features)
    #     return embeddings


    def forward(self, batch_wavefiles):
        if self.frozen:
            with torch.no_grad():
                # features = self.frontend(waveform)['hidden_states'][-1]
                wavlm_features = self.extract_feat_S3PRL(batch_wavefiles)
        else:
            # features = self.frontend(waveform)['hidden_states'][-1]
            wavlm_features = self.extract_feat_S3PRL(batch_wavefiles)



        embeddings = self.embedding(wavlm_features)
        return embeddings

    def extract_feat_S3PRL(self, input_data):
        # Code snipped taken from https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts/blob/master/project/10-asvspoof-vocoded-trn-ssl/model-ID-B1/model.py#L35
        # BSD 3-Clause "New" or "Revised" License
        """ output = extract_feat(input_data)

        input:
        ------
          input_data,tensor, (batch, length, 1) or (batch, length)
          datalength: list of int, length of wav in the mini-batch

        output:
        -------
          output: tensor,
                  if self.layer_indices is None, it has shape
                     (batch, frame_num, frame_feat_dim)
                  else
                     (batch, frame_num, frame_feat_dim),
                     (batch, frame_num, frame_feat_dim, N)
        """

        # put the model to GPU if it not there
        if next(self.frontend.parameters()).device != input_data.device \
                or next(self.frontend.parameters()).dtype != input_data.dtype:
            self.frontend.to(input_data.device, dtype=input_data.dtype)

        # input should be in shape (batch, length)
        if input_data.ndim == 3:
            input_tmp = input_data[:, :, 0]
        else:
            input_tmp = input_data

        input_tmp_len = torch.tensor([x.shape[0] for x in input_data],
                                     dtype=torch.long,
                                     device=input_data.device)

        all_hs, all_hs_len = self.frontend(input_tmp, input_tmp_len)
        emb = all_hs[-1]
        return emb





    @abstractmethod
    def hub_function(self):
        pass
