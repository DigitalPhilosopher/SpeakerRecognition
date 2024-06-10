import librosa
import torch
from torch.utils.data import Dataset
from abc import ABC, abstractmethod


class Loader(Dataset, ABC):
    def __init__(self, directory, frontend, downsampling=0):
        self.frontend = frontend

        self.read_data(directory)

        if downsampling > 0 and len(self.data_list) > downsampling:
            top_speakers = self.data_list['speaker'].value_counts().nlargest(
                downsampling).index
            self.data_list = self.data_list[self.data_list['speaker'].isin(
                top_speakers)].reset_index(drop=True)

        self.genuine = []
        if len(self.data_list) > 0:
            self.genuine = self.data_list[self.data_list["is_genuine"] == 1].reset_index(
                drop=True)

        print(directory)
        print(self.data_list.head())

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
