import torch
from extraction_utils.get_label_files import get_label_files
import time
import numpy as np
import mlflow
import mlflow.pytorch
import gc
import torch.nn.functional as F
import torch.nn as nn
from .distance import compute_distance
from .mining import RandomMiningTrainer, HardMiningTrainer, HardOfflineMiningTrainer, DeepfakeHardOfflineMiningTrainer
import logging
import sys

# Create a logger
logger = logging.getLogger()  # This retrieves the root logger

# Set the logger level
logger.setLevel(logging.INFO)

# Create a console handler and set its level and format
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter(
    'utils/training.py: %(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# Create a file handler and set its level and format
file_handler = logging.FileHandler("utils_training.log")
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Add the handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)


class SemiHardTripletMarginLoss(nn.Module):
    def __init__(self, margin=0.2, distance_function=compute_distance):
        super(SemiHardTripletMarginLoss, self).__init__()
        self.margin = margin
        self.distance_function = distance_function
        self.wrong = 0

    def get_mask(self, p_dist, n_dist):
        return (n_dist < (p_dist + self.margin))

    def forward(self, anchor, positive, negative):
        # Compute pairwise distances
        p_dist = self.distance_function(anchor, positive)
        n_dist = self.distance_function(anchor, negative)

        mask = self.get_mask(p_dist, n_dist)
        self.wrong = (n_dist < p_dist).sum().item()

        # Filter the semi-hard triplets
        p_dist = p_dist[mask]
        n_dist = n_dist[mask]

        # If no semi-hard triplets are found, return a loss of zero
        if p_dist.numel() == 0:
            return torch.tensor(0.0, requires_grad=True).to(anchor.device)

        # Compute the triplet loss only for the semi-hard triplets
        loss = F.relu(p_dist - n_dist + self.margin)

        return loss.mean()


class HardTripletMarginLoss(SemiHardTripletMarginLoss):
    def get_mask(self, p_dist, n_dist):
        p_dist = (p_dist * (1+self.margin))
        return (n_dist < p_dist)


def load_deepfake_dataset(dataset):
    logger.info(f"Loading dataset: {dataset}")
    if dataset == "LibriSpeech":
        return [
            {"name": "clean", "split": "train.100"},
            {"name": "clean", "split": "train.360"},
            {"name": "other", "split": "train.500"}
        ], [
            {"name": "clean", "split": "dev"},
            {"name": "other", "split": "dev"}
        ], [
            {"name": "clean", "split": "test"},
            {"name": "other", "split": "test"}
        ]
    if dataset == "VoxCeleb":
        return [
            {"name": "VoxCeleb2", "split": "dev"},
            {"name": "VoxCeleb1", "split": "dev"},
        ], [
            {"name": "VoxCeleb2", "split": "test"},
            {"name": "VoxCeleb1", "split": "test"},
        ], [
            {"name": "VoxCeleb2", "split": "test"},
            {"name": "VoxCeleb1", "split": "test"},
        ]

    labels_text_path_list_train, labels_text_path_list_dev, labels_text_path_list_test, _ = get_label_files(
        use_bsi_tts=True,
        use_bsi_vocoder=True,
        use_bsi_vc=True,
        use_bsi_genuine=True,
        use_bsi_ttsvctk=False,
        use_bsi_ttslj=False,
        use_bsi_ttsother=False,
        use_bsi_vocoderlj=False,
        use_wavefake=False,
        use_LibriSeVoc=False,
        use_lj=False,
        use_asv2019=False,
    )
    logger.info("Loaded deepfake dataset.")
    return labels_text_path_list_train, labels_text_path_list_dev, labels_text_path_list_test


class ModelTrainer:

    ##### INIT #####

    def __init__(self, model, dataloader, valid_dataloader, test_dataloader,
                 device, loss_function, optimizer, MODEL,
                 FOLDER="Default", TAGS={}, accumulation_steps=1, deepfake=False):
        logger.info("Initializing ModelTrainer.")
        self.model = model
        self.dataloader = dataloader
        self.test_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.MODEL = MODEL
        self.accumulation_steps = accumulation_steps
        self.FOLDER = self.create_or_get_experiment(FOLDER)
        self.TAGS = TAGS
        self.best_loss = float('inf')
        self.best_model_state = None
        self.deepfake = deepfake
        logger.info(f"ModelTrainer initialized with model: {MODEL}")

    def train_epoch(self, epoch, epochs, accumulation_steps=1, triplet_mining="random", create_dataset=None, audio_dataset=None):
        logger.info(
            f"Training epoch {epoch + 1}/{epochs} using {triplet_mining} mining.")
        if triplet_mining == "random":
            return RandomMiningTrainer().train_epoch(epoch, epochs, accumulation_steps, self)
        elif triplet_mining == "hard":
            return HardMiningTrainer().train_epoch(epoch, epochs, accumulation_steps, self)
        elif triplet_mining == "hard-offline":
            if self.deepfake:
                return DeepfakeHardOfflineMiningTrainer().train_epoch(epoch, epochs, accumulation_steps, self, create_dataset, audio_dataset)
            else:
                return HardOfflineMiningTrainer().train_epoch(epoch, epochs, accumulation_steps, self, create_dataset)

    def train_model(self, epochs, start_epoch=1, triplet_mining="random", create_dataset=None, audio_dataset=None):
        logger.info("Starting model training.")
        try:
            mlflow.start_run(run_name=self.MODEL, experiment_id=self.FOLDER)
            self.log_params(epochs)
            self.log_tags()

            if start_epoch != 1:
                logger.info(f"Loading model state from epoch {start_epoch}.")
                self.load_model_state()

            for epoch in range(start_epoch - 1, epochs):
                epoch_start_time = time.time()
                epoch_loss = self.train_epoch(
                    epoch, epochs, accumulation_steps=self.accumulation_steps, triplet_mining=triplet_mining,
                    create_dataset=create_dataset, audio_dataset=audio_dataset)
                avg_loss = epoch_loss / len(self.dataloader)
                self.log_epoch_metrics(avg_loss, epoch_start_time, epoch + 1)

                if avg_loss < self.best_loss:
                    self.best_loss = avg_loss
                    self.best_model_state = self.model.state_dict()
                    self.log_model("best")

                self.save_model_state(epoch)
                gc.collect()

            self.log_model("latest")
            self.save_models()
            logger.info("Model training completed.")

            best_model = self.model
            best_model.load_state_dict(self.best_model_state)
            best_model.to(self.device)
            best_model.eval()
        except Exception as e:
            logger.error("An error occurred during training.", exc_info=True)
        finally:
            mlflow.end_run()
            logger.info("MLflow run ended.")

    def log_params(self, epochs):
        logger.info(f"Logging parameters: {epochs} epochs.")
        mlflow.log_params({
            "Epochs": epochs,
            "Batch size": self.dataloader.batch_size,
            "Model": self.model.__class__.__name__,
            "Loss function": self.loss_function.__class__.__name__,
            "Optimizer": self.optimizer.__class__.__name__
        })

    def log_tags(self):
        logger.info("Logging tags.")
        for key, value in self.TAGS.items():
            mlflow.set_tag(key, value)

    def log_epoch_metrics(self, avg_loss, epoch_start_time, epoch):
        time_minutes = int((time.time() - epoch_start_time) / 60)
        logger.info(
            f"Logging metrics for epoch {epoch}: Average Loss = {avg_loss}, Time = {time_minutes} minutes.")
        mlflow.log_metrics({
            "Average Triplet Loss": avg_loss,
            "Epoch time in minutes": time_minutes
        }, step=epoch)

    def log_model(self, model_type):
        logger.info(f"Logging model: {model_type}.")
        if model_type == "best":
            mlflow.pytorch.log_model(
                self.model, artifact_path=f"{self.MODEL}_best_model_state")
        elif model_type == "latest":
            mlflow.pytorch.log_model(
                self.model, artifact_path=f"{self.MODEL}_latest_model")

    def create_or_get_experiment(self, name):
        logger.info(f"Creating or getting experiment: {name}.")
        experiment = mlflow.get_experiment_by_name(name)
        if experiment:
            return experiment.experiment_id
        else:
            return mlflow.create_experiment(name)

    def save_models(self):
        if self.best_model_state:
            logger.info("Saving best model state.")
            torch.save(self.best_model_state,
                       f"../models/{self.MODEL}_best_model_state.pth")
            mlflow.log_artifact(f"../models/{self.MODEL}_best_model_state.pth")

    def save_model_state(self, epoch):
        logger.info(f"Saving model state for epoch {epoch}.")
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_loss': self.best_loss
        }
        torch.save(state, f'../models/{self.MODEL}_checkpoint.pth')

    def load_model_state(self):
        logger.info("Loading model state from checkpoint.")
        checkpoint = torch.load(f'../models/{self.MODEL}_checkpoint.pth')
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.best_loss = checkpoint['best_loss']
