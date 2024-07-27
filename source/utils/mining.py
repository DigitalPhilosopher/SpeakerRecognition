from abc import ABC, abstractmethod
from tqdm import tqdm
from .distance import l2_normalize, compute_distance
import torch



def hard_chunked_triplet_mining(anchor_embeddings, anchor_labels, device, margin=.2, chunk_size=1000000):
    triplets = []
    all_embeddings = torch.stack(anchor_embeddings).to(device)
    anchor_labels = torch.tensor(anchor_labels)

    for i in tqdm(range(len(anchor_embeddings)), desc="Triplet Mining", leave=True):
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

            chunk_hardest_positive_distance = positive_distances[chunk_hardest_positive_idx].item(
            )
            chunk_hardest_negative_distance = negative_distances[chunk_hardest_negative_idx].item(
            )

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


class MiningEpochTrainer(ABC):
    @abstractmethod
    def train_epoch(self, epoch, epochs, accumulation_steps, modeltrainer, create_dataset=None):
        pass

class RandomMiningTrainer(MiningEpochTrainer):
    def train_epoch(self, epoch, epochs, accumulation_steps, modeltrainer, create_dataset=None):
        return self.train_epoch_triplets(epoch, epochs, accumulation_steps, modeltrainer.dataloader, modeltrainer)

    def train_epoch_triplets(self, epoch, epochs, accumulation_steps, dataloader, modeltrainer):
        modeltrainer.model.train()
        running_loss = 0.0
        last_100_losses = []
        progress_bar = tqdm(
            dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=True)
        for step, (anchors, positives, negatives, metadata) in enumerate(progress_bar):
            try:
                anchors, positives, negatives = anchors.to(
                    modeltrainer.device), positives.to(modeltrainer.device), negatives.to(modeltrainer.device)
                anchor_outputs = modeltrainer.model(anchors)
                positive_outputs = modeltrainer.model(positives)
                negative_outputs = modeltrainer.model(negatives)

                loss = modeltrainer.loss_function(
                    l2_normalize(anchor_outputs), l2_normalize(positive_outputs), l2_normalize(negative_outputs))

                loss.backward()

                if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader):
                    modeltrainer.optimizer.step()
                    modeltrainer.optimizer.zero_grad()
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


class HardOfflineMiningTrainer(RandomMiningTrainer):
    def train_epoch(self, epoch, epochs, accumulation_steps, modeltrainer, create_dataset=None):
        with torch.no_grad():
            embeddings = []
            labels = []
            utterances = []
            running_loss = 0.0

            progress_bar = tqdm(
                modeltrainer.dataloader, desc=f"Epoch {epoch + 1}/{epochs}: Pre mining", leave=True)
            for step, (anchors, _, _, metadata) in enumerate(progress_bar):
                anchors = anchors.to(modeltrainer.device)
                # outputshape: [B, 1, 192]
                anchor_outputs = modeltrainer.model(anchors)

                embeddings += [item for item in anchor_outputs]
                labels += [item["anchor_speaker"] for item in metadata]
                utterances += [item["anchor_utterance"] for item in metadata]

            triplets: torch.LongTensor = hard_chunked_triplet_mining(
                embeddings, labels, modeltrainer.device)  # outputshape: [B, 3]
            if len(triplets) == 0:
                return running_loss

            anchor_utterances = [utterances[i] for i in triplets[:, 0]]
            positive_utterances = [utterances[i] for i in triplets[:, 1]]
            negative_utterances = [utterances[i] for i in triplets[:, 2]]

            training_dataloader = create_dataset(
                anchor_utterances, positive_utterances, negative_utterances)
            torch.cuda.empty_cache()

        return self.train_epoch_triplets(epoch, epochs, accumulation_steps, training_dataloader, modeltrainer)


class HardMiningTrainer(RandomMiningTrainer):
    def train_epoch(self, epoch, epochs, accumulation_steps, modeltrainer, create_dataset=None):
        modeltrainer.model.train()
        running_loss = 0.0
        last_100_losses = []
        progress_bar = tqdm(
            modeltrainer.dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=True)
        for step, (anchors, positives, negatives, metadata) in enumerate(progress_bar):
            anchors, positives, negatives = anchors.to(
                modeltrainer.device), positives.to(modeltrainer.device), negatives.to(modeltrainer.device)
            anchor_outputs = modeltrainer.model(anchors)  # outputshape: [B, 1, 192]
            positive_outputs = modeltrainer.model(
                positives)  # outputshape: [B, 1, 192]
            negative_outputs = modeltrainer.model(
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
                                                                 modeltrainer.device)  # outputshape: [B, 3]
            if len(triplets) == 0:
                continue

            anchor_embeddings = anchor_outputs[triplets[:, 0], ::]
            positive_embeddings = other_embeddings_torch[triplets[:, 1], ::]
            negative_embeddings = other_embeddings_torch[triplets[:, 2], ::]

            loss = modeltrainer.loss_function(
                l2_normalize(anchor_embeddings), l2_normalize(positive_embeddings), l2_normalize(negative_embeddings))

            loss.backward()

            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(modeltrainer.dataloader):
                modeltrainer.optimizer.step()
                modeltrainer.optimizer.zero_grad()
                torch.cuda.empty_cache()

            last_100_losses.append(loss.item())
            if len(last_100_losses) > 100:
                last_100_losses.pop(0)
            running_loss += loss.item()

            progress_bar.set_postfix(loss=loss.item(),
                                     average_loss=sum(last_100_losses) / len(last_100_losses))

        return running_loss