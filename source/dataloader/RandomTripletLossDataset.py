from dataloader import TripletLossDataset


class RandomTripletLossDataset(TripletLossDataset):

    def get_positive(self, anchor_data):
        # Get all genuine samples with the same speaker
        speaker_samples = self.genuine[self.genuine["speaker"]
                                       == anchor_data["speaker"]]

        # Exclude the anchor sample itself
        positive_samples = speaker_samples[speaker_samples["filename"]
                                           != anchor_data["filename"]]
        # Randomly select a positive sample
        positive_data = positive_samples.sample(n=1).iloc[0]
        return positive_data

    def get_negative(self, anchor_data):
        # Get all genuine samples with a different speaker
        negative_samples = self.genuine[self.genuine["speaker"]
                                        != anchor_data["speaker"]]
        # Randomly select a negative sample
        negative_data = negative_samples.sample(n=1).iloc[0]
        return negative_data


class DeepfakeRandomTripletLossDataset(RandomTripletLossDataset):

    def get_negative(self, anchor_data):
        # Get all deepfake samples with a different speaker
        negative_samples = self.data_list[self.data_list["speaker"]
                                          == anchor_data["speaker"]]
        negative_samples = negative_samples[negative_samples["is_genuine"] == 0]
        negative_samples = negative_samples[negative_samples["method_type"] != "Vocoder"]
        # Randomly select a negative sample
        negative_data = negative_samples.sample(n=1).iloc[0]
        return negative_data
    
class DeepfakeRandomTripletLoss_VocoderPositiveDataset(DeepfakeRandomTripletLossDataset):

    def get_positive(self, anchor_data):
        # Get all deepfake samples with a different speaker
        positive_samples = self.data_list[self.data_list["speaker"]
                                          == anchor_data["speaker"]]
        positive_samples = positive_samples[positive_samples["method_type"] == "Vocoder"]
        # Randomly select a negative sample
        positive_data = positive_samples.sample(n=1).iloc[0]
        return positive_data

    def get_negative(self, anchor_data):
        # Get all deepfake samples with a different speaker
        negative_samples = self.data_list[self.data_list["speaker"]
                                          == anchor_data["speaker"]]
        negative_samples = negative_samples[negative_samples["is_genuine"] == 0]
        negative_samples = negative_samples[negative_samples["method_type"] != "Vocoder"]
        # Randomly select a negative sample
        negative_data = negative_samples.sample(n=1).iloc[0]
        return negative_data


class DeepfakeRandomTripletLossSameUtteranceDataset(RandomTripletLossDataset):

    def get_negative(self, anchor_data):
        # Get all deepfake samples with a different speaker
        negative_samples = self.data_list[self.data_list["speaker"]
                                          == anchor_data["speaker"]]
        negative_samples = negative_samples[negative_samples["is_genuine"] == 0]
        negative_samples = negative_samples[negative_samples["method_type"] != "Vocoder"]
        negative_samples = negative_samples[negative_samples["utterance"] == anchor_data["utterance"]]
        # Randomly select a negative sample
        negative_data = negative_samples.sample(n=1).iloc[0]
        return negative_data
