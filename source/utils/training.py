import torch
from extraction_utils.get_label_files import get_label_files
from tqdm import tqdm
import time
import torch.nn.functional as F
import mlflow
import mlflow.pytorch
from .validation import ModelValidator
import gc
import numpy as np
from .distance import l2_normalize, compute_distance
from dataloader import HardTripletLossDataset


def load_deepfake_dataset(dataset):
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
        use_bsi_vocoder=False,
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
    return labels_text_path_list_train, labels_text_path_list_dev, labels_text_path_list_test

def hard_chunked_triplet_mining(anchor_embeddings, anchor_labels, device, margin=.2, chunk_size=1000):
    triplets = []
    all_embeddings = torch.stack(anchor_embeddings).to(device)
    anchor_labels = torch.tensor(anchor_labels)

    for i in range(len(anchor_embeddings)):
        anchor = anchor_embeddings[i].to(device)
        anchor_label = anchor_labels[i].item()

        positive_mask = (anchor_labels == anchor_label).to(device)
        negative_mask = (anchor_labels != anchor_label).to(device)

        if positive_mask.sum().item() <= 1:
            continue

        num_chunks = (all_embeddings.size(0) + chunk_size - 1) // chunk_size
    
        hardest_positive_distance = float('-inf')
        hardest_positive_idx = -1
        
        hardest_negative_distance = float('inf')
        hardest_negative_idx = -1

        for j in range(num_chunks):
            start_idx = j * chunk_size
            end_idx = min((j + 1) * chunk_size, all_embeddings.size(0))
            
            chunk = all_embeddings[start_idx:end_idx]
            
            distances = compute_distance(anchor.unsqueeze(0), chunk)
            
            chunk_positive_mask = positive_mask[start_idx:end_idx]
            chunk_negative_mask = negative_mask[start_idx:end_idx]
            
            positive_distances = torch.where(chunk_positive_mask, distances,
                                            torch.tensor(float('-inf')).to(device))
            negative_distances = torch.where(chunk_negative_mask, distances,
                                            torch.tensor(float('inf')).to(device))
            
            chunk_hardest_positive_idx = positive_distances.argmax()
            chunk_hardest_negative_idx = negative_distances.argmin()
            
            chunk_hardest_positive_distance = positive_distances[chunk_hardest_positive_idx].item()
            chunk_hardest_negative_distance = negative_distances[chunk_hardest_negative_idx].item()
            
            if chunk_hardest_positive_distance > hardest_positive_distance:
                hardest_positive_distance = chunk_hardest_positive_distance
                hardest_positive_idx = start_idx + chunk_hardest_positive_idx.item()
            
            if chunk_hardest_negative_distance < hardest_negative_distance:
                hardest_negative_distance = chunk_hardest_negative_distance
                hardest_negative_idx = start_idx + chunk_hardest_negative_idx.item()

        if hardest_positive_idx != -1 and hardest_negative_idx != -1:
            triplets.append([i, hardest_positive_idx, hardest_negative_idx])

    return torch.LongTensor(triplets)


def hard_triplet_mining(anchor_embeddings, anchor_labels, combined_embeddings, combined_labels, device, margin=.2):
    triplets = []

    for i in range(len(anchor_embeddings)):
        anchor = anchor_embeddings[i]
        anchor_label = anchor_labels[i]

        positive_mask = (combined_labels == anchor_label)
        negative_mask = (combined_labels != anchor_label)

        if len(positive_mask) == 1:
            continue

        distances = compute_distance(anchor.unsqueeze(
            0), torch.stack(combined_embeddings))

        positive_distances = torch.where(torch.from_numpy(positive_mask).to(device), distances,
                                         torch.tensor(float('-inf')))
        negative_distances = torch.where(torch.from_numpy(negative_mask).to(device), distances,
                                         torch.tensor(float('inf')))

        hardest_positive_idx = positive_distances.argmax()
        hardest_negative_idx = negative_distances.argmin()

        triplets.append([i, hardest_positive_idx.item(),
                        hardest_negative_idx.item()])

    return torch.LongTensor(triplets)


class ModelTrainer:

    ##### INIT #####

    def __init__(self, model, dataloader, valid_dataloader, test_dataloader,
                 device, loss_function, optimizer, MODEL,
                 FOLDER="Default", TAGS={}, accumulation_steps=1, validation_rate=5):
        self.model = model
        self.dataloader = dataloader
        self.test_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.validation_rate = validation_rate
        self.device = device
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.MODEL = MODEL
        self.accumulation_steps = accumulation_steps
        self.FOLDER = self.create_or_get_experiment(FOLDER)
        self.TAGS = TAGS
        self.best_loss = float('inf')
        self.best_model_state = None
        self.validator = ModelValidator(valid_dataloader, device)

    ##### TRAINING #####

    def train_epoch(self, epoch, epochs, accumulation_steps=1, triplet_mining="random", create_dataset=None):
        if triplet_mining == "random":
            return self.train_epoch_random(epoch, epochs, accumulation_steps)

        elif triplet_mining == "hard":
            return self.train_epoch_hard(epoch, epochs, accumulation_steps)

        elif triplet_mining == "hard-offline":
            return self.train_epoch_hard_offline(epoch, epochs, accumulation_steps, create_dataset)

    def train_epoch_hard(self, epoch, epochs, accumulation_steps):
        self.model.train()
        running_loss = 0.0
        last_100_losses = []
        progress_bar = tqdm(
            self.dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=True)
        for step, (anchors, positives, negatives, metadata) in enumerate(progress_bar):
            anchors, positives, negatives = anchors.to(
                self.device), positives.to(self.device), negatives.to(self.device)
            anchor_outputs = self.model(anchors)  # outputshape: [B, 1, 192]
            positive_outputs = self.model(
                positives)  # outputshape: [B, 1, 192]
            negative_outputs = self.model(
                negatives)  # outputshape: [B, 1, 192]

            other_embeddings_torch = torch.cat(
                [anchor_outputs, positive_outputs, negative_outputs], dim=0)
            with torch.no_grad():
                embeddings = [item for item in anchor_outputs]
                labels = [item["anchor_speaker"] for item in metadata]
                other_embeddings = embeddings + [item for item in positive_outputs] + [item for item in
                                                                                       negative_outputs]
                other_labels = labels + [item["positive_speaker"] for item in metadata] + [item["negative_speaker"] for
                                                                                           item in metadata]
                triplets: torch.LongTensor = hard_triplet_mining(embeddings, labels, other_embeddings, other_labels,
                                                                 self.device)  # outputshape: [B, 3]
            if len(triplets) == 0:
                continue

            anchor_embeddings = anchor_outputs[triplets[:, 0], ::]
            positive_embeddings = other_embeddings_torch[triplets[:, 1], ::]
            negative_embeddings = other_embeddings_torch[triplets[:, 2], ::]

            loss = self.loss_function(
                l2_normalize(anchor_embeddings), l2_normalize(positive_embeddings), l2_normalize(negative_embeddings))

            loss.backward()

            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(self.dataloader):
                self.optimizer.step()
                self.optimizer.zero_grad()
                torch.cuda.empty_cache()

            last_100_losses.append(loss.item())
            if len(last_100_losses) > 100:
                last_100_losses.pop(0)
            running_loss += loss.item()

            progress_bar.set_postfix(loss=loss.item(),
                                     average_loss=sum(last_100_losses) / len(last_100_losses))

        return running_loss

    def train_epoch_hard_offline(self, epoch, epochs, accumulation_steps, create_dataset=None):
        with torch.no_grad():
            embeddings = []
            labels = []
            utterances = []
            running_loss = 0.0

            progress_bar = tqdm(
                self.dataloader, desc=f"Epoch {epoch + 1}/{epochs}: Pre mining", leave=True)
            for step, (anchors, _, _, metadata) in enumerate(progress_bar):
                anchors = anchors.to(self.device)
                anchor_outputs = self.model(anchors)  # outputshape: [B, 1, 192]

                embeddings += [item for item in anchor_outputs]
                labels += [item["anchor_speaker"] for item in metadata]
                utterances += [item["anchor_utterance"] for item in metadata]
            
            triplets: torch.LongTensor = hard_chunked_triplet_mining(embeddings, labels, self.device)  # outputshape: [B, 3]
            if len(triplets) == 0:
                return running_loss

            anchor_utterances = [utterances[i] for i in triplets[:, 0]]
            positive_utterances = [utterances[i] for i in triplets[:, 1]]
            negative_utterances = [utterances[i] for i in triplets[:, 2]]

            training_dataloader = create_dataset(
                anchor_utterances, positive_utterances, negative_utterances)
            torch.cuda.empty_cache()

        return self.train_epoch_triplets(epoch, epochs, accumulation_steps, training_dataloader)

    def train_epoch_random(self, epoch, epochs, accumulation_steps):
        return self.train_epoch_triplets(epoch, epochs, accumulation_steps, self.dataloader)

    def train_epoch_triplets(self, epoch, epochs, accumulation_steps, dataloader):
        self.model.train()
        running_loss = 0.0
        last_100_losses = []
        progress_bar = tqdm(
            dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=True)
        for step, (anchors, positives, negatives, metadata) in enumerate(progress_bar):
            try:
                anchors, positives, negatives = anchors.to(
                    self.device), positives.to(self.device), negatives.to(self.device)
                anchor_outputs = self.model(anchors)
                positive_outputs = self.model(positives)
                negative_outputs = self.model(negatives)

                loss = self.loss_function(
                    l2_normalize(anchor_outputs), l2_normalize(positive_outputs), l2_normalize(negative_outputs))

                loss.backward()

                if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    torch.cuda.empty_cache()

                last_100_losses.append(loss.item())
                if len(last_100_losses) > 100:
                    last_100_losses.pop(0)
                running_loss += loss.item()

                progress_bar.set_postfix(loss=loss.item(),
                                         average_loss=sum(last_100_losses) / len(last_100_losses))

            except Exception as e:
                print(f"Error during training: {e}")
                torch.cuda.empty_cache()  # Clear cache in case of error
                continue

        return running_loss

    def train_model(self, epochs, start_epoch=1, triplet_mining="random", create_dataset=None):
        try:
            mlflow.start_run(run_name=self.MODEL, experiment_id=self.FOLDER)
            self.log_params(epochs)
            self.log_tags()

            if start_epoch != 1:
                self.load_model_state()

            for epoch in range(start_epoch - 1, epochs):
                epoch_start_time = time.time()
                epoch_loss = self.train_epoch(
                    epoch, epochs, accumulation_steps=self.accumulation_steps, triplet_mining=triplet_mining,
                    create_dataset=create_dataset)
                avg_loss = epoch_loss / len(self.dataloader)
                self.log_epoch_metrics(avg_loss, epoch_start_time, epoch + 1)

                if avg_loss < self.best_loss:
                    self.best_loss = avg_loss
                    self.best_model_state = self.model.state_dict()
                    self.log_model("best")

                self.save_model_state(epoch)
                if (epoch + 1) % self.validation_rate == 0:
                    try:
                        self.validator.validate_model(self.model, epoch + 1)
                    except Exception as e:
                        print(f"Error during validation: {e}")

                gc.collect()

            self.log_model("latest")
            self.save_models()

            best_model = self.model
            best_model.load_state_dict(self.best_model_state)
            best_model.to(self.device)
            best_model.eval()
            tester = ModelValidator(self.test_dataloader, self.device)
            tester.validate_model(best_model, epochs, "Best Model - ")
        finally:
            mlflow.end_run()

    ##### LOGGING #####

    def log_params(self, epochs):
        mlflow.log_params({
            "Epochs": epochs,
            "Batch size": self.dataloader.batch_size,
            "Model": self.model.__class__.__name__,
            "Loss function": self.loss_function.__class__.__name__,
            "Optimizer": self.optimizer.__class__.__name__
        })

    def log_tags(self):
        for key, value in self.TAGS.items():
            mlflow.set_tag(key, value)

    def log_epoch_metrics(self, avg_loss, epoch_start_time, epoch):
        time_minutes = int((time.time() - epoch_start_time) / 60)
        mlflow.log_metrics({
            "Average Triplet Loss": avg_loss,
            "Epoch time in minutes": time_minutes
        }, step=epoch)

    def log_model(self, model_type):
        if model_type == "best":
            mlflow.pytorch.log_model(
                self.model, artifact_path=f"{self.MODEL}_best_model_state")
        elif model_type == "latest":
            mlflow.pytorch.log_model(
                self.model, artifact_path=f"{self.MODEL}_latest_model")

    def create_or_get_experiment(self, name):
        experiment = mlflow.get_experiment_by_name(name)
        if experiment:
            return experiment.experiment_id
        else:
            return mlflow.create_experiment(name)

    ##### SAVING #####

    def save_models(self):
        if self.best_model_state:
            torch.save(self.best_model_state,
                       f"../models/{self.MODEL}_best_model_state.pth")
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
        checkpoint = torch.load(
            f'../models/{self.MODEL}_checkpoint.pth')
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.best_loss = checkpoint['best_loss']
