import os
import torchaudio
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


class AudioDataset(Dataset):
    def __init__(self, directory, frontend):
        self.directory = directory
        self.frontend = frontend
        self.audio_files = []
        self.labels = []
        self.speaker_id_to_label = {}

        # Generate a list of files and corresponding labels
        current_label = 0
        for dirpath, _, filenames in os.walk(directory):
            for filename in filenames:
                if filename.endswith('.wav'):
                    self.audio_files.append(os.path.join(dirpath, filename))
                    speaker_id = os.path.basename(dirpath)
                    if speaker_id not in self.speaker_id_to_label:
                        self.speaker_id_to_label[speaker_id] = current_label
                        current_label += 1
                    self.labels.append(self.speaker_id_to_label[speaker_id])

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        waveform, sample_rate = torchaudio.load(audio_path)

        frontend_call = self.frontend(sample_rate=sample_rate, number_output_parameters = 13)
        frontend_embedding = frontend_call(waveform)

        frontend_embedding = frontend_embedding.squeeze(0).transpose(0, 1)  # Remove batch dimension and transpose

        label = self.labels[idx]
        return frontend_embedding, label

def collate_fn(batch):
    frontend_embeddings, speaker_ids = zip(*batch)
    target_time_steps = 1024

    max_feature_dim = max(frontend_embedding.size(1) for frontend_embedding in frontend_embeddings)

    frontend_embeddings_processed = []
    for frontend_embedding in frontend_embeddings:
        # Ensure all tensors have the same feature dimension (pad if necessary)
        if frontend_embedding.size(1) < max_feature_dim:
            padding = max_feature_dim - frontend_embedding.size(1)
            frontend_embedding = F.pad(frontend_embedding, (0, padding, 0, 0))  # Pad feature dimension

        # Pad or truncate the time dimension
        if frontend_embedding.size(0) > target_time_steps:
            frontend_embedding = frontend_embedding[:target_time_steps, :]
        else:
            pad_amount = target_time_steps - frontend_embedding.size(0)
            frontend_embedding = F.pad(frontend_embedding, (0, 0, 0, pad_amount))  # Pad time dimension

        frontend_embeddings_processed.append(frontend_embedding)

    frontend_embeddings_padded = torch.stack(frontend_embeddings_processed)  # [batch, time, features]
    labels = torch.tensor(speaker_ids, dtype=torch.long)

    return frontend_embeddings_padded, labels
