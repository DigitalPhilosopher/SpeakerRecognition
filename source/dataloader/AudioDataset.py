import os
from typing import List, Tuple
import librosa
import pandas as pd
import torch
from torch.utils.data import Dataset
from extraction_utils.data_utils import read_label_file
from torch.nn.utils.rnn import pad_sequence

class AudioDataset(Dataset):
    def __init__(self, directory, frontend, logger):
        self.frontend = frontend
        self.logger = logger

        self.data_list: List[Tuple[str, int]] = read_label_file(directory)
        self.data_list = pd.DataFrame(self.data_list, columns=["filename", "is_genuine", "method_type", "method_name", "vocoder"])

        self.data_list["utterance"] = self.data_list["filename"].apply(lambda x: os.path.basename(x).split(".")[0])
        self.data_list["speaker"] = self.data_list["utterance"].apply(lambda x: x.split("_")[0])

        self.genuine = self.data_list[self.data_list["is_genuine"] == 1].reset_index(drop=True)

        num_speakers = self.data_list["speaker"].nunique()
        num_utterances = len(self.data_list)
        num_genuine_utterances = len(self.genuine)
        num_deepfake_utterances = num_utterances - num_genuine_utterances

        logger.info(f"Number of speakers: {num_speakers}")
        logger.info(f"Number of utterances: {num_utterances}")
        logger.info(f"Number of genuine utterances: {num_genuine_utterances}")
        logger.info(f"Number of deepfake utterances: {num_deepfake_utterances}")

        # TODO: Remove
        # Group by speaker and select top 5 speakers
        # top_speakers = self.data_list['speaker'].value_counts().nlargest(5).index
        # sampled_data = self.data_list[self.data_list['speaker'].isin(top_speakers)].reset_index(drop=True)
        # self.data_list = sampled_data
        # self.genuine = self.data_list[self.data_list["is_genuine"] == 1].reset_index(drop=True)
        # logger.warn("Downsampled to have only 5 speakers, keeping all utterances")


    def __len__(self):
        return len(self.genuine)

    def __getitem__(self, idx):
        anchor_data = self.genuine.iloc[idx]
        return self.read_audio(anchor_data["filename"])
    
    def read_audio(self, filename):
        waveform, sample_rate = librosa.load(filename, sr=16000) # Read wav
        waveform, _ = librosa.effects.trim(waveform, top_db=35) # Remove silence at beginning and end of wav

        waveform = torch.tensor(waveform, dtype=torch.float32)

        return self.frontend(waveform)

def collate_triplet_fn(batch):
    anchors, positives, negatives = zip(*batch)
    anchors = torch.stack(anchors)
    positives = torch.stack(positives)
    negatives = torch.stack(negatives)
    return anchors, positives, negatives

def collate_triplet_wav_fn(batch):
    anchors, positives, negatives = zip(*batch)

    # Pad sequences directly without additional squeezing since tensors are already 2D
    anchors = pad_sequence(anchors, batch_first=True, padding_value=0)
    positives = pad_sequence(positives, batch_first=True, padding_value=0)
    negatives = pad_sequence(negatives, batch_first=True, padding_value=0)

    return anchors, positives, negatives
