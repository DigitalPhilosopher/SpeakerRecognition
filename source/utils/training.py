import torch
from extraction_utils.get_label_files import get_label_files
from tqdm.notebook import tqdm
import time
import numpy as np
import mlflow
import mlflow.pytorch
from sklearn.metrics import roc_curve
from itertools import combinations


def load_genuine_dataset():
    labels_text_path_list_train, labels_text_path_list_dev, labels_text_path_list_test, _ = get_label_files(
        use_bsi_tts = False,
        use_bsi_vocoder = False,
        use_bsi_vc = False,
        use_bsi_genuine = True,
        use_bsi_ttsvctk = False,
        use_bsi_ttslj = False,
        use_bsi_ttsother = False,
        use_bsi_vocoderlj = False,
        use_wavefake = False,
        use_LibriSeVoc = False,
        use_lj = False,
        use_asv2019 = False,
    )
    return labels_text_path_list_train, labels_text_path_list_dev, labels_text_path_list_test


def load_deepfake_dataset():
    labels_text_path_list_train, labels_text_path_list_dev, labels_text_path_list_test, _ = get_label_files(
        use_bsi_tts = True,
        use_bsi_vocoder = False,
        use_bsi_vc = False,
        use_bsi_genuine = True,
        use_bsi_ttsvctk = False,
        use_bsi_ttslj = False,
        use_bsi_ttsother = False,
        use_bsi_vocoderlj = False,
        use_wavefake = False,
        use_LibriSeVoc = False,
        use_lj = False,
        use_asv2019 = False,
    )
    return labels_text_path_list_train, labels_text_path_list_dev, labels_text_path_list_test


class ModelTrainer:

    ##### INIT #####

    def __init__(self, model, dataloader, valid_dataloader, device, loss_function, optimizer, logger, MODEL, FOLDER="Default", validation_rate=5):
        self.model = model
        self.dataloader = dataloader
        self.valid_dataloader = valid_dataloader
        self.validation_rate = validation_rate
        self.device = device
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.logger = logger
        self.MODEL = MODEL
        self.FOLDER = FOLDER
        self.best_loss = float('inf')
        self.best_model_state = None


    ##### TRAINING #####

    def train_epoch(self, epoch, epochs):
        self.model.train()
        running_loss = 0.0
        progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
        for anchors, positives, negatives in progress_bar:
            try:
                anchors, positives, negatives = anchors.to(self.device), positives.to(self.device), negatives.to(self.device)
                self.optimizer.zero_grad()

                anchor_outputs = self.model(anchors)
                positive_outputs = self.model(positives)
                negative_outputs = self.model(negatives)

                loss = self.loss_function(anchor_outputs, positive_outputs, negative_outputs)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())
            except Exception as e:
                self.logger.error(f"Error during training: {e}")
                continue
        return running_loss


    def train_model(self, epochs, start_epoch=1):
        try:
            mlflow.start_run(run_name=self.MODEL, experiment_id=self.FOLDER)
            self.log_params(epochs)
            total_start_time = time.time()

            if start_epoch != 0:
                self.load_model_state()

            for epoch in range(start_epoch, epochs):
                epoch_start_time = time.time()
                epoch_loss = self.train_epoch(epoch, epochs)
                avg_loss = epoch_loss / len(self.dataloader)
                self.log_epoch_metrics(avg_loss, epoch_start_time, epoch)

                if avg_loss < self.best_loss:
                    self.best_loss = avg_loss
                    self.best_model_state = self.model.state_dict()
                    self.log_model("best")

                if (epoch + 1) % self.validation_rate == 0:
                    eer, min_dcf = self.validate_model(self.valid_dataloader)
                    self.logger.info(f'Validation EER: {eer:.4f}, minDCF: {min_dcf:.4f}')
                    mlflow.log_metrics({'validation_eer': eer, 'validation_min_dcf': min_dcf}, step=epoch)

                self.save_model_state(epoch)

            self.log_model("latest")
            self.save_models()
            self.logger.info(f"Training completed in {time.time() - total_start_time:.4f} seconds.")
        finally:
            mlflow.end_run()


    ##### VALIDATION #####

    def validate_model(self, dataloader):
        self.model.eval()
        embeddings = []
        labels = []
        with torch.no_grad():
            for data in dataloader:
                inputs, targets = data
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                embeddings.extend(outputs.data.cpu().numpy())
                labels.extend(targets)

        scores, score_labels = self.pairwise_scores(embeddings, labels)
        eer, min_dcf = self.compute_metrics(scores, score_labels)
        return eer, min_dcf


    def pairwise_scores(self, embeddings, labels):
        scores = []
        score_labels = []
        # Compute pairwise scores
        for (emb1, lbl1), (emb2, lbl2) in combinations(zip(embeddings, labels), 2):
            # Ensure that embeddings are 1D
            emb1 = np.squeeze(emb1)
            emb2 = np.squeeze(emb2)
            score = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))  # Cosine similarity
            scores.append(score)
            score_labels.append(1 if lbl1 == lbl2 else 0)
        return scores, score_labels


    def compute_metrics(self, scores, score_labels, c_fa=1, c_fr=1, p_target=0.01):
        # Calculate the EER
        fpr, tpr, thresholds = roc_curve(score_labels, scores, pos_label=1)
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.abs(fpr - fnr))]
        
        # Calculate the minDCF
        min_dcf = float('inf')
        for threshold in thresholds:
            fnr_at_threshold = 1 - tpr[thresholds >= threshold][0]
            fpr_at_threshold = fpr[thresholds >= threshold][0]
            dcf = c_fr * fnr_at_threshold * p_target + c_fa * fpr_at_threshold * (1 - p_target)
            if dcf < min_dcf:
                min_dcf = dcf
        
        return eer, min_dcf


    ##### LOGGING #####

    def log_params(self, epochs):
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": self.dataloader.batch_size,
            "model": self.model.__class__.__name__,
            "loss_function": self.loss_function.__class__.__name__,
            "optimizer": self.optimizer.__class__.__name__
        })


    def log_epoch_metrics(self, avg_loss, epoch_start_time, epoch):
        mlflow.log_metrics({"avg_loss": avg_loss, "epoch_time": time.time() - epoch_start_time}, step=epoch)


    def log_model(self, model_type):
        if model_type == "best":
            mlflow.pytorch.log_model(self.model, artifact_path=f"{self.MODEL}_best_model_state")
        elif model_type == "latest":
            mlflow.pytorch.log_model(self.model, artifact_path=f"{self.MODEL}_latest_model")


    ##### SAVING #####

    def save_models(self):
        if self.best_model_state:
            torch.save(self.best_model_state, f"../models/{self.MODEL}_best_model_state.pth")
            mlflow.log_artifact(f"../models/{self.MODEL}_best_model_state.pth")


    def save_model_state(self, epoch):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_loss': self.best_loss
        }
        torch.save(state, f'../models/{self.MODEL}_checkpoint.pth')


    def load_model_state(self):
        checkpoint = torch.load(f'../models/{self.MODEL}_checkpoint.pth')
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.best_loss = checkpoint['best_loss']