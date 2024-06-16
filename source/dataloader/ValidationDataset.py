from dataloader import AudioDataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np


class ValidationDataset(AudioDataset):
    def __getitem__(self, idx):
        anchor_data = self.data_list.iloc[idx]
        return (self.read_audio(anchor_data["filename"]), anchor_data["speaker"], anchor_data["utterance"], anchor_data["is_genuine"], anchor_data["method_name"])

    def __len__(self):
        return len(self.data_list)


def collate_valid_fn(batch):
    audio_data, speakers, utterances, is_genuine, method_name = zip(*batch)
    audio_data = pad_sequence(audio_data, batch_first=True, padding_value=0)
    return audio_data, np.array(speakers), np.array(utterances), np.array(is_genuine), np.array(method_name)
