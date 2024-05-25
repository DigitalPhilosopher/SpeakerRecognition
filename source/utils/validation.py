import torch
import time
import numpy as np
import mlflow
import mlflow.pytorch
from collections import defaultdict
from sklearn.metrics import roc_curve
from itertools import combinations
import gc
from tqdm import tqdm


def compute_distance(emb1, emb2):
    emb1 = np.squeeze(emb1)
    emb2 = np.squeeze(emb2)
    return np.linalg.norm(emb1 - emb2)


class ModelValidator:

    ##### INIT #####

    def __init__(self, valid_dataloader, device):
        self.dataloader = valid_dataloader
        self.device = device

    ##### VALIDATION #####
    def validate_model(self, model, step=-1, speaker_eer=True, deepfake_eer=True, mlflow_logging=True, prefix=""):
        start_time = time.time()

        sv_eer, sv_threshold, dd_eer, dd_threshold = -1, -1, -1, -1

        model.eval()

        embeddings, labels, deepfake_embeddings, deepfake_labels, deepfake_methods = self.generate_embeddings(
            model)

        if speaker_eer:
            scores, score_labels = self.pairwise_scores(embeddings, labels)
            sv_eer, sv_threshold = self.compute_eer(scores, score_labels)

            if mlflow_logging:
                mlflow.log_metrics({
                    prefix + 'EER - Speaker Verification': sv_eer,
                    prefix + 'Threshold - Speaker Verification': sv_threshold,
                }, step=step)

        if deepfake_eer:
            genuine_deepfake_scores, genuine_deepfake_labels, method_scores = self.generate_deepfake_pairwise_scores(
                labels, embeddings, deepfake_labels, deepfake_embeddings, deepfake_methods)
            dd_eer, dd_threshold = self.compute_eer(
                genuine_deepfake_scores, genuine_deepfake_labels)

            # Find the hardest method
            avg_method_scores = {method: np.mean(
                scores) for method, scores in method_scores.items()}
            hardest_method = min(avg_method_scores, key=avg_method_scores.get)
            hardest_method_score = avg_method_scores[hardest_method]

            if mlflow_logging:
                mlflow.log_metrics({
                    prefix + 'EER - Deepfake Detection': dd_eer,
                    prefix + 'Threshold - Deepfake Detection': dd_threshold,
                    prefix + 'Hardest Deepfake Method': hardest_method_score
                }, step=step)

        end_time = time.time()
        validation_time_minutes = int((end_time - start_time) / 60)
        if mlflow_logging:
            mlflow.log_metrics({
                prefix + 'Validation time in minutes': validation_time_minutes
            }, step=step)

        gc.collect()

        return sv_eer, sv_threshold, dd_eer, dd_threshold

    def generate_embeddings(self, model):
        embeddings = []
        labels = []

        deepfake_embeddings = []
        deepfake_labels = []
        deepfake_methods = []

        with torch.no_grad():
            for data in tqdm(self.dataloader, desc="Generating embeddings"):
                inputs, targets, is_genuine, method = data
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                embedding_output = outputs.data.cpu().numpy()
                genuine_indices = is_genuine == 1

                # Separate embeddings and labels for genuine and deepfake
                embeddings.extend(embedding_output[genuine_indices])
                labels.extend(targets[genuine_indices])
                deepfake_embeddings.extend(embedding_output[~genuine_indices])
                deepfake_labels.extend(targets[~genuine_indices])
                deepfake_methods.extend(method[~genuine_indices])
        return embeddings, labels, deepfake_embeddings, deepfake_labels, deepfake_methods

    def generate_deepfake_pairwise_scores(self, labels, embeddings, deepfake_labels, deepfake_embeddings, deepfake_methods):
        unique_labels = list(set(labels))

        genuine_deepfake_scores = []
        genuine_deepfake_labels = []
        method_scores = defaultdict(list)

        for speaker in tqdm(unique_labels, desc="Calculating deepfake distances"):
            genuine_indices = [i for i, lbl in enumerate(
                labels) if lbl == speaker]
            deepfake_indices = [i for i, lbl in enumerate(
                deepfake_labels) if lbl == speaker]

            # Compute scores for genuine-genuine pairs
            for gi1, gi2 in combinations(genuine_indices, 2):
                genuine_deepfake_scores.append(
                    compute_distance(embeddings[gi1], embeddings[gi2]))
                # 1 indicates genuine-genuine
                genuine_deepfake_labels.append(1)

            # Compute scores for genuine-deepfake pairs
            for gi in genuine_indices:
                for di in deepfake_indices:
                    score = compute_distance(
                        embeddings[gi], deepfake_embeddings[di])
                    genuine_deepfake_scores.append(score)
                    genuine_deepfake_labels.append(0)
                    method_scores[deepfake_methods[di]].append(score)

        return genuine_deepfake_scores, genuine_deepfake_labels, method_scores

    def pairwise_scores(self, embeddings, labels):
        scores = []
        score_labels = []
        # Compute pairwise scores
        for (emb1, lbl1), (emb2, lbl2) in tqdm(combinations(zip(embeddings, labels), 2), desc="Computing pairwise scores"):
            scores.append(compute_distance(emb1, emb2))
            score_labels.append(1 if lbl1 == lbl2 else 0)
        return np.array(scores), np.array(score_labels)

    def compute_eer(self, scores, score_labels):
        # Calculate the EER
        fpr, tpr, thresholds = roc_curve(score_labels, scores, pos_label=1)
        fnr = 1 - tpr

        threshold = thresholds[np.nanargmin(np.abs(fnr - fpr))]
        eer = fpr[np.nanargmin(np.abs(fnr - fpr))]

        return eer, threshold
