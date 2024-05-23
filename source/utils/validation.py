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

    def __init__(self, valid_dataloader, device):
        self.dataloader = valid_dataloader
        self.device = device


    ##### VALIDATION #####

    def validate_model(self, model, step):
        start_time = time.time()
        model.eval()

        embeddings = []
        labels = []
        
        deepfake_embeddings = []
        deepfake_labels = []

        with torch.no_grad():
            for data in self.dataloader:
                inputs, targets, is_genuine = data
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                embedding_output = outputs.data.cpu().numpy() 
                genuine_indices = is_genuine == 1

                # Separate embeddings and labels for genuine and deepfake
                embeddings.extend(embedding_output[genuine_indices])
                labels.extend(targets[genuine_indices])
                deepfake_embeddings.extend(embedding_output[~genuine_indices])
                deepfake_labels.extend(targets[~genuine_indices])

        scores, score_labels = self.pairwise_scores(embeddings, labels)
        eer, threshold = self.compute_eer(scores, score_labels)

        mlflow.log_metrics({
            'EER - Speaker Verification': eer,
            'Threshold - Speaker Verification': threshold,
            }, step=step)

        unique_labels = list(set(labels))
    
        genuine_deepfake_scores = []
        genuine_deepfake_labels = []

        for speaker in unique_labels:
            genuine_indices = [i for i, lbl in enumerate(labels) if lbl == speaker]
            deepfake_indices = [i for i, lbl in enumerate(deepfake_labels) if lbl == speaker]

            # Compute scores for genuine-genuine pairs
            for gi1, gi2 in combinations(genuine_indices, 2):
                emb1 = np.squeeze(embeddings[gi1])
                emb2 = np.squeeze(embeddings[gi2])
                score = np.linalg.norm(emb1 - emb2)
                genuine_deepfake_scores.append(score)
                genuine_deepfake_labels.append(1)  # 1 indicates genuine-genuine

            # Compute scores for genuine-deepfake pairs
            for gi in genuine_indices:
                for di in deepfake_indices:
                    emb1 = np.squeeze(embeddings[gi])
                    emb2 = np.squeeze(deepfake_embeddings[di])
                    score = np.linalg.norm(emb1 - emb2)
                    genuine_deepfake_scores.append(score)
                    genuine_deepfake_labels.append(0)
        
        eer, threshold = self.compute_eer(genuine_deepfake_scores, genuine_deepfake_labels)

        end_time = time.time()
        validation_time_minutes = int((end_time - start_time) / 60)
        mlflow.log_metrics({
            'EER - Deepfake Detection': eer,
            'Threshold - Deepfake Detection': threshold,
            'Validation time in minutes': validation_time_minutes
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