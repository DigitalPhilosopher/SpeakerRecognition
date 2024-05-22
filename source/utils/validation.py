import torch
from extraction_utils.get_label_files import get_label_files
from tqdm import tqdm
import time
import numpy as np
import mlflow
import mlflow.pytorch
from sklearn.metrics import roc_curve
from itertools import combinations


class ModelValidator:

    ##### INIT #####

    def __init__(self,valid_dataloader, device):
        self.dataloader = valid_dataloader
        self.device = device


    ##### VALIDATION #####

    def validate_model(self, model, step):
        model.eval()
        embeddings = []
        labels = []
        with torch.no_grad():
            for data in self.dataloader:
                inputs, targets = data
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                embeddings.extend(outputs.data.cpu().numpy())
                labels.extend(targets)

        scores, score_labels = self.pairwise_scores(embeddings, labels)
        eer, threshold = self.compute_eer(scores, score_labels)

        mlflow.log_metrics({
            'EER - Speaker Verification': eer,
            'Threshold - Speaker Verification': threshold,
            }, step=step)


    def pairwise_scores(self, embeddings, labels):
        scores = []
        score_labels = []
        # Compute pairwise scores
        for (emb1, lbl1), (emb2, lbl2) in combinations(zip(embeddings, labels), 2):
            # Ensure that embeddings are 1D
            emb1 = np.squeeze(emb1)
            emb2 = np.squeeze(emb2)
            score = np.linalg.norm(emb1 - emb2)
            scores.append(score)
            score_labels.append(1 if lbl1 == lbl2 else 0)
        return np.array(scores), np.array(score_labels)


    def compute_eer(self, scores, score_labels):
        # Calculate the EER
        fpr, tpr, thresholds = roc_curve(score_labels, scores, pos_label=1)
        fnr = 1 - tpr

        threshold = thresholds[np.nanargmin(np.abs(fnr - fpr))]
        eer = fpr[np.nanargmin(np.abs(fnr - fpr))]

        return eer, threshold