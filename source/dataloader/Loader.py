import librosa
import torch
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
import pandas as pd


class Loader(Dataset, ABC):
    def __init__(self, directory, frontend, downsampling=0, max_audios=0):
        self.frontend = frontend

        self.read_data(directory)

        if downsampling > 0 and len(self.data_list) > downsampling:
            top_speakers = self.data_list['speaker'].value_counts().nlargest(
                downsampling).index
            self.data_list = self.data_list[self.data_list['speaker'].isin(
                top_speakers)].reset_index(drop=True)
        if max_audios > 0:
            # Only use "max_audios" genuine audios per speaker
            genuine_entries = self.data_list[self.data_list['is_genuine'] == True]

            filtered_genuine = genuine_entries.groupby('speaker').apply(
                lambda x: x.head(max_audios) if len(x) > max_audios else x
            ).reset_index(drop=True)
            non_genuine_entries = self.data_list[self.data_list['is_genuine'] == False]
            self.data_list = pd.concat([filtered_genuine, non_genuine_entries]).reset_index(drop=True)

        self.genuine = []
        if len(self.data_list) > 0:
            self.genuine = self.data_list[self.data_list["is_genuine"] == 1].reset_index(
                drop=True)


    @ abstractmethod
    def read_data(self, directory):
        pass

    def get_data(self):
        return (self.genuine, self.data_list)

    def read_audio(self, filename):
        waveform, _ = librosa.load(filename, sr=16000)
        waveform, _ = librosa.effects.trim(waveform, top_db=35)
        waveform = torch.tensor(waveform, dtype=torch.float32)
        return self.frontend(waveform)
