import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class AudioDataset(Dataset):
    def __init__(self, loader):

        self.loader = loader
        self.data_list, self.genuine = loader.get_data()

    def __len__(self):
        return len(self.genuine)

    def __getitem__(self, idx):
        anchor_data = self.genuine.iloc[idx]
        return self.read_audio(anchor_data["filename"])

    def read_audio(self, filename):
        return self.loader.read_audio(filename)


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
