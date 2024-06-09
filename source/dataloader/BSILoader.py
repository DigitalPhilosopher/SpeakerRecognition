import os
from typing import List, Tuple
import librosa
import pandas as pd
import torch
from torch.utils.data import Dataset
from extraction_utils.data_utils import read_label_file


def extract_speaker_id(file_path):
    dir_name = os.path.dirname(file_path)
    dir_parts = dir_name.split(os.path.sep)
    last_folder = dir_parts[-1]
    speaker_id = last_folder.split("_")[-1]
    return speaker_id


def read_audio(filename, frontend=lambda x: x):
    waveform, _ = librosa.load(filename, sr=16000)
    waveform, _ = librosa.effects.trim(waveform, top_db=35)
    waveform = torch.tensor(waveform, dtype=torch.float32)
    return frontend(waveform)


class BSILoader(Dataset):
    def __init__(self, directory, frontend, downsampling=0):
        self.frontend = frontend

        self.data_list: List[Tuple[str, int]] = read_label_file(directory)
        self.data_list = pd.DataFrame(self.data_list, columns=[
                                      "filename", "is_genuine", "method_type", "method_name", "vocoder"])

        self.data_list["utterance"] = self.data_list["filename"].apply(
            lambda x: os.path.basename(x).split(".")[0])
        self.data_list["speaker"] = self.data_list["filename"].apply(
            extract_speaker_id)

        if downsampling > 0:
            top_speakers = self.data_list['speaker'].value_counts().nlargest(
                downsampling).index
            self.data_list = self.data_list[self.data_list['speaker'].isin(
                top_speakers)].reset_index(drop=True)

        self.genuine = self.data_list[self.data_list["is_genuine"] == 1].reset_index(
            drop=True)

    def get_data(self):
        return (self.genuine, self.data_list)

    def read_audio(self, filename):
        return read_audio(filename, self.frontend)
