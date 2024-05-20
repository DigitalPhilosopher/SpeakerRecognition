import torch
from extraction_utils.get_label_files import get_label_files
from tqdm.notebook import tqdm
import time
import mlflow
import mlflow.pytorch


def train_model_random_loss(epochs, dataloader, model, loss_function, optimizer, device, MODEL, logger):
    latest_model_name = f"{MODEL}_latest_model"
    best_model_name = f"{MODEL}_best_model_state"

    best_loss = float('inf')
    best_model_state = None

    with mlflow.start_run(run_name=MODEL):
        # Logging model and training details
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": dataloader.batch_size,
            "model": model.__class__.__name__,
            "loss_function": loss_function.__class__.__name__,
            "optimizer": optimizer.__class__.__name__
        })

        model.train()
        total_start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            running_loss = 0.0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)

            for anchors, positives, negatives in progress_bar:
                try:
                    anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)
                    optimizer.zero_grad()
                    
                    anchor_outputs = model(anchors)
                    positive_outputs = model(positives)
                    negative_outputs = model(negatives)
                    
                    loss = loss_function(anchor_outputs, positive_outputs, negative_outputs)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    progress_bar.set_postfix(loss=loss.item())

                except Exception as e:
                    logger.error(f"Error during training: {e}")
                    continue

            avg_loss = running_loss / len(dataloader)
            mlflow.log_metrics({"avg_loss": avg_loss, "epoch_time": time.time() - epoch_start_time}, step=epoch)

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = model.state_dict()
                mlflow.pytorch.log_model(model, artifact_path=best_model_name)

        mlflow.log_metric("total_training_time", time.time() - total_start_time)
        mlflow.pytorch.log_model(model, artifact_path=latest_model_name)

        if best_model_state:
            torch.save(best_model_state, f"../models/{MODEL}_best_model_state.pth")
            mlflow.log_artifact(f"../models/{MODEL}_best_model_state.pth")

        logger.info(f"Training completed in {time.time() - total_start_time:.4f} seconds.")


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
    def __init__(self, model, dataloader, device, loss_function, optimizer, logger, MODEL):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.logger = logger
        self.MODEL = MODEL
        self.best_loss = float('inf')
        self.best_model_state = None

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

    def train_model(self, epochs):
        mlflow.start_run(run_name=self.MODEL)
        self.log_params(epochs)
        total_start_time = time.time()

        for epoch in range(epochs):
            epoch_start_time = time.time()
            epoch_loss = self.train_epoch(epoch, epochs)
            avg_loss = epoch_loss / len(self.dataloader)
            self.log_epoch_metrics(avg_loss, epoch_start_time, epoch)

            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.best_model_state = self.model.state_dict()
                self.log_model("best")

        self.log_model("latest")
        self.save_models()
        self.logger.info(f"Training completed in {time.time() - total_start_time:.4f} seconds.")

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

    def save_models(self):
        if self.best_model_state:
            torch.save(self.best_model_state, f"../models/{self.MODEL}_best_model_state.pth")
            mlflow.log_artifact(f"../models/{self.MODEL}_best_model_state.pth")
