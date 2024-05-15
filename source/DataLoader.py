import os
from typing import List, Tuple
import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from source.extraction_utils.data_utils import read_label_file

class AudioDataset(Dataset):
    def __init__(self, directory, frontend):
        self.frontend = frontend

        self.data_list: List[Tuple[str, int]] = read_label_file(directory)
        self.data_list = pd.DataFrame(self.data_list, columns=["filename", "is_genuine", "method_type", "method_name", "vocoder"])

        self.data_list["utterance"] = self.data_list["filename"].apply(lambda x: os.path.basename(x).split(".")[0])
        self.data_list["speaker"] = self.data_list["utterance"].apply(lambda x: x.split("_")[0])

        self.genuine = self.data_list[self.data_list["is_genuine"] == 1].reset_index(drop=True)

    def __len__(self):
        return len(self.genuine)

    def __getitem__(self, idx):
        anchor_data = self.genuine.iloc[idx]
        positive_data = self.get_positive(anchor_data)
        negative_data = self.get_negative(anchor_data)
        return self.get_triplet(anchor_data, positive_data, negative_data)
    
    def get_triplet(self, anchor_data, negative_data, positive_data):
        anchor = self.read_audio(anchor_data["filename"])
        positive = self.read_audio(negative_data["filename"])
        negative = self.read_audio(positive_data["filename"])
        return anchor, positive, negative

    def get_positive(self, anchor_data):
        # TODO
        return anchor_data

    def get_negative(self, anchor_data):
        # TODO
        return anchor_data
    
    def read_audio(self, filename):
        waveform, sample_rate = librosa.load(filename, sr=16000) # Read wav
        waveform, _ = librosa.effects.trim(waveform, top_db=35) # Remove silence at beginning and end of wav

        waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)

        mfcc_transform = self.frontend(number_output_parameters=13, sample_rate=sample_rate)
        return mfcc_transform(waveform)

def collate_fn(batch):
    anchors, positives, negatives = zip(*batch)
    return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)
