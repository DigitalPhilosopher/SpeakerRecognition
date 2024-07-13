from dataloader import TripletLossDataset


class HardTripletLossDataset(TripletLossDataset):

    def set_triplets(self, anchors, positives, negatives):
        self.anchors = anchors
        self.positives = positives
        self.negatives = negatives

    def get_positive(self, anchor_data):
        utterance = anchor_data["utterance"]

        if utterance not in self.anchors:
            return self.random_positive(anchor_data)

        idx = self.anchors.index(utterance)

        putterance = self.positives[idx]
        pidx = self.genuine[self.genuine['utterance'] == putterance].index[0]

        positive_data = self.genuine.iloc[pidx]
        return positive_data

    def get_negative(self, anchor_data):
        utterance = anchor_data["utterance"]

        if utterance not in self.anchors:
            return self.random_negative(anchor_data)

        idx = self.anchors.index(utterance)

        nutterance = self.negatives[idx]
        nidx = self.genuine[self.genuine['utterance'] == nutterance].index[0]

        negative_data = self.genuine.iloc[nidx]
        return negative_data

    def random_positive(self, anchor_data):
        # Get all genuine samples with the same speaker
        speaker_samples = self.genuine[self.genuine["speaker"]
                                       == anchor_data["speaker"]]

        # Exclude the anchor sample itself
        positive_samples = speaker_samples[speaker_samples["filename"]
                                           != anchor_data["filename"]]
        # Randomly select a positive sample
        positive_data = positive_samples.sample(n=1).iloc[0]
        return positive_data

    def random_negative(self, anchor_data):
        # Get all genuine samples with a different speaker
        negative_samples = self.genuine[self.genuine["speaker"]
                                        != anchor_data["speaker"]]
        # Randomly select a negative sample
        negative_data = negative_samples.sample(n=1).iloc[0]
        return negative_data
