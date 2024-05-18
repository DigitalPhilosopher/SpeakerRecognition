import os
from typing import List, Tuple
import librosa
import numpy as np
import pandas as pd
import torch
import logging
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

        num_speakers = self.data_list["speaker"].nunique()
        num_utterances = len(self.data_list)
        num_genuine_utterances = len(self.genuine)
        num_deepfake_utterances = num_utterances - num_genuine_utterances

        logger = logging.getLogger()
        logger.info(f"Number of speakers: {num_speakers}")
        logger.info(f"Number of utterances: {num_utterances}")
        logger.info(f"Number of genuine utterances: {num_genuine_utterances}")
        logger.info(f"Number of deepfake utterances: {num_deepfake_utterances}")

        # TODO: Remove
        # Downsample to have only 5 speakers with 2 utterances each
        sampled_data = (
            self.data_list.groupby("speaker")
            .apply(lambda x: x.sample(2) if len(x) >= 2 else None)
            .dropna()
            .reset_index(drop=True)
        )

        if len(sampled_data["speaker"].unique()) > 5:
            sampled_speakers = sampled_data["speaker"].unique()[:5]
            sampled_data = sampled_data[sampled_data["speaker"].isin(sampled_speakers)]

        self.data_list = sampled_data.reset_index(drop=True)
        self.genuine = self.data_list[self.data_list["is_genuine"] == 1].reset_index(drop=True)
        logger.warn("Downsample to have only 5 speakers with 2 utterances each")

    def __len__(self):
        return len(self.genuine)

    def __getitem__(self, idx):
        anchor_data = self.genuine.iloc[idx]
        return self.read_audio(anchor_data["filename"])
    
    def read_audio(self, filename):
        waveform, sample_rate = librosa.load(filename, sr=16000) # Read wav
        waveform, _ = librosa.effects.trim(waveform, top_db=35) # Remove silence at beginning and end of wav

        waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)

        frontend = self.frontend(number_output_parameters=13, sample_rate=sample_rate)
        return frontend(waveform).squeeze(0).permute(1, 0)

def collate_single_fn(batch):
    pass

def collate_triplet_fn(batch):
    anchors, positives, negatives = zip(*batch)
    anchors = torch.stack(anchors)
    positives = torch.stack(positives)
    negatives = torch.stack(negatives)
    return anchors, positives, negatives