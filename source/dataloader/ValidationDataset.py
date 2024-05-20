from dataloader import AudioDataset
from torch.nn.utils.rnn import pad_sequence

class ValidationDataset(AudioDataset):
    def __getitem__(self, idx):
        anchor_data = self.genuine.iloc[idx]
        return (self.read_audio(anchor_data["filename"]), anchor_data["speaker"])

def collate_valid_fn(batch):
    audio_data, speakers = zip(*batch)
    audio_data = pad_sequence(audio_data, batch_first=True, padding_value=0)
    return audio_data, speakers
