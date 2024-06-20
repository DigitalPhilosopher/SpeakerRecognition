import os
import torch
from torch.nn import TripletMarginWithDistanceLoss
import time
import numpy as np
import pandas as pd
import mlflow
import mlflow.pytorch
from collections import defaultdict
from sklearn.metrics import roc_curve
from itertools import combinations
import gc
from tqdm import tqdm
from .distance import compute_distance, l2_normalize


def get_valid_sets(name):
    valid_set = []
    df_valid_set = []

    if name == "LibriSpeech":
        valid_set = pd.read_csv("../validation_sets/LibriSpeech/valid.csv")

    return valid_set, df_valid_set


def get_train_sets(name):
    valid_set = []
    df_valid_set = []

    if name == "LibriSpeech":
        valid_set = pd.read_csv("../validation_sets/LibriSpeech/train.csv")

    return valid_set, df_valid_set


def get_test_sets(name):
    valid_set = []
    df_valid_set = []

    if name == "LibriSpeech":
        valid_set = pd.read_csv("../validation_sets/LibriSpeech/test.csv")

    return valid_set, df_valid_set


class ModelValidator:

    ##### INIT #####

    def __init__(self, valid_dataloader, device, valid_set=[], df_valid_set=[]):
        self.dataloader = valid_dataloader
        self.device = device
        self.valid_set = valid_set
        self.df_valid_set = df_valid_set

    ##### VALIDATION #####
    def validate_model(self, model, step=-1, speaker_eer=True, deepfake_eer=True, mlflow_logging=True, prefix=""):
        start_time = time.time()

        sv_eer, sv_threshold, dd_eer, dd_threshold = -1, -1, -1, -1
        sv_rates, dd_rates = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}, {
            "TP": 0, "TN": 0, "FP": 0, "FN": 0}

        sv_min_dcf, dd_min_dcf = -1, -1

        if not self.dataloader:
            return sv_eer, sv_threshold, sv_rates, sv_min_dcf, dd_eer, dd_threshold, dd_rates, dd_min_dcf

        model.eval()

        embeddings, labels, utterances, deepfake_embeddings, deepfake_labels, deepfake_utterances, deepfake_methods = self.generate_embeddings(
            model)

        if speaker_eer:
            if len(self.valid_set) > 0:
                scores, score_labels = self.pairwise_scores_with_set(
                    embeddings, utterances, self.valid_set)
            else:
                scores, score_labels = self.pairwise_scores(embeddings, labels)
                average_loss = self.pariwise_loss(embeddings, labels)
                print(f"!!!!!!!!!!!!!!!average_loss: {average_loss}")
                scores_genuine = [scores[i] for i in range(len(scores)) if score_labels[i] == 1]
                scores_imposter = [scores[i] for i in range(len(scores)) if score_labels[i] == 0]
                print("!!!!!scores_genuine:", sum(scores_genuine)/len(scores_genuine))
                print("!!!!!scores_imposter:", sum(scores_imposter)/len(scores_imposter))
                from source.utils.plot_score_lists import plot_similarity_lists_bar, calc_eer
                plot_similarity_lists_bar(
                    [scores_imposter, scores_genuine],
                    ["False Accept attempt", "Genuine"], do_plot=False,
                    save_plot_path=os.path.join("..", "logs", "score_plot.png"))
                eer = calc_eer(scores_genuine, scores_imposter)
                print("!!!!!eer:", eer)

            sv_eer, sv_threshold = self.compute_eer(scores, score_labels)
            sv_min_dcf = self.compute_min_dcf(scores, score_labels)
            TP, TN, FP, FN = self.compute_tp_tn_fp_fn(
                scores, score_labels, sv_threshold)
            sv_rates = {
                "TP": TP,
                "TN": TN,
                "FP": FP,
                "FN": FN,
            }

            if mlflow_logging:
                mlflow.log_metrics({
                    prefix + 'EER - Speaker Verification': sv_eer,
                    prefix + 'Threshold - Speaker Verification': sv_threshold,
                    prefix + 'minDCF - Speaker Verification': sv_min_dcf
                }, step=step)

        if deepfake_eer and len(deepfake_embeddings) > 0:
            genuine_deepfake_scores, genuine_deepfake_labels, method_scores = self.generate_deepfake_pairwise_scores(
                labels, embeddings, deepfake_labels, deepfake_embeddings, deepfake_methods)
            dd_eer, dd_threshold = self.compute_eer(
                genuine_deepfake_scores, genuine_deepfake_labels)
            dd_min_dcf = self.compute_min_dcf(
                genuine_deepfake_scores, genuine_deepfake_labels)
            TP, TN, FP, FN = self.compute_tp_tn_fp_fn(
                genuine_deepfake_scores, genuine_deepfake_labels, dd_threshold)
            dd_rates = {
                "TP": TP,
                "TN": TN,
                "FP": FP,
                "FN": FN,
            }

            # Find the hardest method
            avg_method_scores = {method: np.mean(
                scores) for method, scores in method_scores.items()}
            hardest_method = min(avg_method_scores, key=avg_method_scores.get)
            hardest_method_score = avg_method_scores[hardest_method]

            if mlflow_logging:
                mlflow.log_metrics({
                    prefix + 'EER - Deepfake Detection': dd_eer,
                    prefix + 'Threshold - Deepfake Detection': dd_threshold,
                    prefix + 'minDCF - Deepfake Detection': dd_min_dcf,
                    prefix + 'Hardest Deepfake Method': hardest_method_score
                }, step=step)

        end_time = time.time()
        validation_time_minutes = int((end_time - start_time) / 60)
        if mlflow_logging:
            mlflow.log_metrics({
                prefix + 'Validation time in minutes': validation_time_minutes
            }, step=step)

        gc.collect()

        return sv_eer, sv_threshold, sv_rates, sv_min_dcf, dd_eer, dd_threshold, dd_rates, dd_min_dcf

    def generate_embeddings(self, model):
        embeddings = []
        labels = []
        utterances = []

        deepfake_embeddings = []
        deepfake_labels = []
        deepfake_utterances = []
        deepfake_methods = []

        with torch.no_grad():
            for data in tqdm(self.dataloader, desc="Generating embeddings"):
                inputs, targets, utterance_ids, is_genuine, method = data
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                embedding_output = outputs.data.cpu()
                genuine_indices = is_genuine == 1

                # Separate embeddings and labels for genuine and deepfake
                embeddings.extend(embedding_output[genuine_indices])
                labels.extend(targets[genuine_indices])
                utterances.extend(utterance_ids[genuine_indices])

                deepfake_embeddings.extend(embedding_output[~genuine_indices])
                deepfake_labels.extend(targets[~genuine_indices])
                deepfake_utterances.extend(utterance_ids[~genuine_indices])
                deepfake_methods.extend(method[~genuine_indices])
        return embeddings, labels, utterances, deepfake_embeddings, deepfake_labels, deepfake_utterances, deepfake_methods

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
                    compute_distance(l2_normalize(embeddings[gi1]), l2_normalize(embeddings[gi2])))
                # 1 indicates genuine-genuine
                genuine_deepfake_labels.append(1)

            # Compute scores for genuine-deepfake pairs
            for gi in genuine_indices:
                for di in deepfake_indices:
                    score = compute_distance(
                        l2_normalize(embeddings[gi]), l2_normalize(deepfake_embeddings[di]))
                    genuine_deepfake_scores.append(score)
                    genuine_deepfake_labels.append(0)
                    method_scores[deepfake_methods[di]].append(score)

        return genuine_deepfake_scores, genuine_deepfake_labels, method_scores

    def pariwise_loss(self, embeddings, labels, valid_set=[]) -> float:
        triplet_loss = TripletMarginWithDistanceLoss(
            distance_function=compute_distance, margin=0.2)

        losses = []
        # Compute pairwise scores
        for (emb1, lbl1), (emb2, lbl2), (emb3, lbl3) in tqdm(combinations(zip(embeddings, labels), 3), desc="Compute average loss"):
            if lbl1 == lbl2 and lbl1 != lbl3:
                loss = triplet_loss(
                    l2_normalize(emb1), l2_normalize(emb2),
                    l2_normalize(emb3))
                losses.append(loss)
        return sum(losses) / len(losses)

    def pairwise_scores(self, embeddings, labels, valid_set=[]):
        scores = []
        score_labels = []
        # Compute pairwise scores
        for (emb1, lbl1), (emb2, lbl2) in tqdm(combinations(zip(embeddings, labels), 2), desc="Computing pairwise scores"):
            scores.append(compute_distance(
                l2_normalize(emb1), l2_normalize(emb2)))
            score_labels.append(1 if lbl1 == lbl2 else 0)
        return np.array(scores), np.array(score_labels)

    def pairwise_scores_with_set(self, embeddings, utterances, pairs_df):
        scores = []
        score_labels = []

        # Create a dictionary for fast lookup of embeddings by utterance ID
        embedding_dict = dict(zip(utterances, embeddings))
        # Loop through each row in the pairs_df
        for _, row in tqdm(pairs_df.iterrows(), desc="Computing pairwise scores", total=pairs_df.shape[0]):
            if row['utterance'] not in embedding_dict or row['utterance_to_check'] not in embedding_dict:
                continue
            emb1 = embedding_dict[row['utterance']]
            emb2 = embedding_dict[row['utterance_to_check']]
            scores.append(compute_distance(
                l2_normalize(emb1), l2_normalize(emb2)))
            score_labels.append(row['is_same_speaker'])
        return np.array(scores), np.array(score_labels)

    def compute_eer(self, scores, score_labels):
        # Calculate the EER
        fpr, tpr, thresholds = roc_curve(score_labels, scores, pos_label=1)
        fnr = 1 - tpr

        threshold = thresholds[np.nanargmin(np.abs(fnr - fpr))]
        eer = fpr[np.nanargmin(np.abs(fnr - fpr))]

        return eer, threshold

    def compute_tp_tn_fp_fn(self, scores, score_labels, threshold):
        # Initialize counters
        TP = TN = FP = FN = 0

        # Classify scores based on the threshold
        predicted_labels = scores >= threshold

        # Calculate TP, TN, FP, and FN
        for true_label, predicted_label in zip(score_labels, predicted_labels):
            if true_label == 1 and predicted_label == 1:
                TP += 1
            elif true_label == 0 and predicted_label == 0:
                TN += 1
            elif true_label == 0 and predicted_label == 1:
                FP += 1
            elif true_label == 1 and predicted_label == 0:
                FN += 1

        return TP, TN, FP, FN

    def compute_min_dcf(self, scores, score_labels, p_target=0.01, c_miss=1, c_fa=1):
        fpr, tpr, thresholds = roc_curve(score_labels, scores, pos_label=1)
        fnr = 1 - tpr

        # Define the function for DCF
        def dcf(beta, fnr, fpr):
            return c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)

        # Compute the minDCF
        min_dcf = float('inf')
        for fpr_i, fnr_i in zip(fpr, fnr):
            dcf_i = dcf(1, fnr_i, fpr_i)
            if dcf_i < min_dcf:
                min_dcf = dcf_i

        return min_dcf
