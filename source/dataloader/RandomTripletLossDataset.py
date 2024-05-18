from source.dataloader import TripletLossDataset

class RandomTripletLossDataset(TripletLossDataset):
    
    def get_positive(self, anchor_data):
        # Get all genuine samples with the same speaker
        speaker_samples = self.genuine[self.genuine["speaker"] == anchor_data["speaker"]]
        # Exclude the anchor sample itself
        positive_samples = speaker_samples[speaker_samples["filename"] != anchor_data["filename"]]
        # Randomly select a positive sample
        positive_data = positive_samples.sample(n=1).iloc[0]
        return positive_data

    def get_negative(self, anchor_data):
        # Get all genuine samples with a different speaker
        negative_samples = self.genuine[self.genuine["speaker"] != anchor_data["speaker"]]
        # Randomly select a negative sample
        negative_data = negative_samples.sample(n=1).iloc[0]
        return negative_data
