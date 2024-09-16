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


class RandomTripletLossDatasetDeepfakeAnchor(RandomTripletLossDataset):
    def __init__(self, loader, max_length: int = 0):
        super().__init__(loader, max_length)

        speaker_counts = self.genuine['speaker'].value_counts()
        self.data_list = self.data_list[self.data_list['speaker'].map(
            speaker_counts) >= 2].reset_index(drop=True)

        self.deepfakes_list = self.data_list[(self.data_list["is_genuine"] == 0) & (
            self.data_list["method_type"] != "Vocoder")]

    def __len__(self):
        return len(self.deepfakes_list)

    def get_positives(self, negative_data):
        speaker_samples = self.genuine[self.genuine["speaker"]
                                       == negative_data["speaker"]]

        samples = speaker_samples.sample(n=2)
        anchor_data = samples.iloc[0]
        positive_data = samples.iloc[1]

        return anchor_data, positive_data

    def __getitem__(self, idx):
        negative_data = self.deepfakes_list.iloc[idx]
        anchor_data, positive_data = self.get_positives(negative_data)
        return self.get_triplet(anchor_data, positive_data, negative_data)


class RandomTripletLossDatasetDeepfakeAnchor_Vocoder(RandomTripletLossDatasetDeepfakeAnchor):
    def get_positives(self, negative_data):
        speaker_samples = self.genuine[self.genuine["speaker"]
                                       == negative_data["speaker"]]

        samples = speaker_samples.sample(n=1)
        anchor_data = samples.iloc[0]

        positive_samples = self.data_list[self.data_list["speaker"]
                                          == anchor_data["speaker"]]
        positive_samples = positive_samples[positive_samples["method_type"] == "Vocoder"]
        # Randomly select a negative sample
        positive_data = positive_samples.sample(n=1).iloc[0]

        return anchor_data, positive_data


class DeepfakeRandomTripletLossDataset(RandomTripletLossDataset):

    def __init__(self, loader, max_length: int = 0):
        super(DeepfakeRandomTripletLossDataset,
              self).__init__(loader, max_length)
        self.method_type = None

    def set_method(self, method):
        self.method_type = method

    def get_negative(self, anchor_data):
        # Get all deepfake samples with a different speaker
        negative_samples = self.data_list[self.data_list["speaker"]
                                          == anchor_data["speaker"]]
        negative_samples = negative_samples[negative_samples["is_genuine"] == 0]

        negative_samples = negative_samples[negative_samples["method_type"] != "Vocoder"]
        if self.method_type:
            _negative_samples = negative_samples[negative_samples["method_name"]
                                                 == self.method_type]
            if len(_negative_samples) > 0:
                negative_samples = _negative_samples

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
        negative_samples = negative_samples[negative_samples["utterance"]
                                            == anchor_data["utterance"]]
        # Randomly select a negative sample
        negative_data = negative_samples.sample(n=1).iloc[0]
        return negative_data
