from dataloader import TripletLossDataset


class HardTripletLossDataset(TripletLossDataset):

    def set_triplets(self, anchors, positives, negatives):
        self.anchors = anchors
        self.positives = positives
        self.negatives = negatives

    def get_positive(self, anchor_data):
        utterance = anchor_data["utterance"]
        idx = self.anchors.index(utterance)

        putterance = self.positives[idx]
        pidx = self.genuine[self.genuine['utterance'] == putterance].index[0]

        positive_data = self.genuine.iloc[pidx]
        return positive_data

    def get_negative(self, anchor_data):
        utterance = anchor_data["utterance"]
        idx = self.anchors.index(utterance)

        nutterance = self.negatives[idx]
        nidx = self.genuine[self.genuine['utterance'] == nutterance].index[0]

        negative_data = self.genuine.iloc[nidx]
        return negative_data

