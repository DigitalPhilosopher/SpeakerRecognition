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
        speaker, filename = self.get_speaker_and_filename(audio_path)

        return self.get_triplet(audio_path, self.get_positive(speaker, filename), self.get_negative(speaker))
    
    def get_speaker_and_filename(self, audio_path):
        filename = os.path.basename(audio_path)
        speaker = filename.split('_')[0]
        return speaker, filename

    def get_positive(self, speaker, filename):
        for file in self.audio_files:
            s, f = self.get_speaker_and_filename(file)
            if s == speaker:
                if not f == filename:
                    return file

    def get_negative(self, speaker):
        for file in self.audio_files:
            s, _ = self.get_speaker_and_filename(file)
            if not s == speaker:
                return file
    
    def get_triplet(self, anchor_audio_path, positive_audio_path, negative_audio_path):
        anchor = self.getitem(anchor_audio_path)
        positive = self.getitem(positive_audio_path)
        negative = self.getitem(negative_audio_path)
        
        return anchor, positive, negative

    def getitem(self, audio_path):
        waveform, sample_rate = torchaudio.load(audio_path)

        frontend_call = self.frontend(sample_rate=sample_rate, number_output_parameters = 13)
        frontend_embedding = frontend_call(waveform)

        frontend_embedding = frontend_embedding.squeeze(0).transpose(0, 1)  # Remove batch dimension and transpose

        return frontend_embedding

def collate_fn(batch):
    anchors, positives, negatives = zip(*batch)
    return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)
