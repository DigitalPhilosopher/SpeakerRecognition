from abc import ABC, abstractmethod
from source.dataloader import AudioDataset

class TripletLossDataset(AudioDataset, ABC):
    
    @abstractmethod
    def get_positive(self, anchor_data):
        """Abstract method to get a positive sample."""
        pass

    @abstractmethod
    def get_negative(self, anchor_data):
        """Abstract method to get a negative sample."""
        pass

    def __getitem__(self, idx):
        anchor_data = self.genuine.iloc[idx]
        positive_data = self.get_positive(anchor_data)
        negative_data = self.get_negative(anchor_data)
        return self.get_triplet(anchor_data, positive_data, negative_data)

    def get_triplet(self, anchor_data, positive_data, negative_data):
        anchor = self.read_audio(anchor_data["filename"])
        positive = self.read_audio(positive_data["filename"])
        negative = self.read_audio(negative_data["filename"])
        return anchor, positive, negative
