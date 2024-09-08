import logging
import sys
from tqdm import tqdm
import torch
import random
from abc import ABC, abstractmethod
from .distance import l2_normalize, compute_distance

# Create a logger
logger = logging.getLogger()  # This retrieves the root logger

# Set the logger level
logger.setLevel(logging.DEBUG)

# Create a console handler and set its level and format
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_formatter = logging.Formatter(
    'utils/mining.py%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# Create a file handler and set its level and format
file_handler = logging.FileHandler("utils_mining.log")
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Add the handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)


def hard_chunked_triplet_mining(anchor_embeddings, anchor_labels, device, margin=.2, chunk_size=1000000):
    logger.info("Starting hard chunked triplet mining.")
    triplets = []
    all_embeddings = torch.stack(anchor_embeddings).to(device)
    anchor_labels = torch.tensor(anchor_labels)

    for i in tqdm(range(len(anchor_embeddings)), desc="Triplet Mining", leave=True):
        anchor = anchor_embeddings[i].to(device)
        anchor_label = anchor_labels[i].item()

        positive_mask = (anchor_labels == anchor_label).to(device)
        negative_mask = (anchor_labels != anchor_label).to(device)

        if positive_mask.sum().item() <= 1:
            logger.debug(
                f"Skipping anchor {i} due to insufficient positive samples.")
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

    logger.info("Completed hard chunked triplet mining.")
    return torch.LongTensor(triplets)


def hard_triplet_mining(anchor_embeddings, anchor_labels, combined_embeddings, combined_labels, device, margin=.2):
    logger.info("Starting hard triplet mining.")
    triplets = []

    for i in range(len(anchor_embeddings)):
        anchor = anchor_embeddings[i]
        anchor_label = anchor_labels[i]

        positive_mask = (combined_labels == anchor_label)
        negative_mask = (combined_labels != anchor_label)

        if len(positive_mask) == 1:
            logger.debug(
                f"Skipping anchor {i} due to insufficient positive samples.")
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

    logger.info("Completed hard triplet mining.")
    return torch.LongTensor(triplets)


class MiningEpochTrainer(ABC):
    @abstractmethod
    def train_epoch(self, epoch, epochs, accumulation_steps, modeltrainer, create_dataset=None):
        pass


class RandomMiningTrainer(MiningEpochTrainer):
    def train_epoch(self, epoch, epochs, accumulation_steps, modeltrainer, create_dataset=None):
        logger.info(f"Starting epoch {epoch + 1}/{epochs} with random mining.")
        return self.train_epoch_triplets(epoch, epochs, accumulation_steps, modeltrainer.dataloader, modeltrainer)

    def train_epoch_triplets(self, epoch, epochs, accumulation_steps, dataloader, modeltrainer):
        modeltrainer.model.train()
        running_loss = 0.0
        running_total = 0
        running_wrong = 0
        last_100_losses = []
        progress_bar = tqdm(
            dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=True)
        batch_size = dataloader.batch_size
        from sklearn.metrics import roc_curve
        import numpy as np

        def get_eer_for_two_distance_lists(genuine_list, spoof_list):
            label_list = [0] * len(genuine_list) + [1] * len(spoof_list)
            fpr, tpr, thresholds = roc_curve(
                label_list, genuine_list + spoof_list)
            fnr = 1 - tpr
            eer_index = np.nanargmin(np.absolute(fpr - fnr))
            eer = fpr[eer_index]
            eer_threshold = thresholds[eer_index]
            return eer

        positive_distance_list = []
        negative_distance_list = []
        for step, (anchors, positives, negatives, metadata) in enumerate(progress_bar):
            try:
                # modeltrainer.model.eval()

                anchors, positives, negatives = anchors.to(modeltrainer.device), positives.to(
                    modeltrainer.device), negatives.to(modeltrainer.device)
                DO_PRINT = False
                # anchors = anchors[:2, :]
                # positives = positives[:2, :]
                # negatives = negatives[:2, :]

                # anchor_outputs = modeltrainer.model(anchors)
                # positive_outputs = modeltrainer.model(positives)
                # negative_outputs = modeltrainer.model(negatives)

                all_inputs = torch.cat([anchors, positives, negatives], dim=0)
                all_outputs = modeltrainer.model(all_inputs)

                anchor_outputs = all_outputs[:anchors.shape[0], ::]
                positive_outputs = all_outputs[anchors.shape[0]
                    : 2*anchors.shape[0], ::]
                negative_outputs = all_outputs[2 * anchors.shape[0]:, ::]

                # anchor_outputs = anchor_outputs[:2, ::]
                # positive_outputs = positive_outputs[:2, ::]
                # negative_outputs = negative_outputs[:2, ::]

                p_dist = modeltrainer.loss_function.distance_function(
                    (anchor_outputs), (positive_outputs))
                n_dist = modeltrainer.loss_function.distance_function(
                    (anchor_outputs), (negative_outputs))

                # TODO, test without normalization anchor_norm = l2_normalize(anchor_outputs)
                # positive_outputs = l2_normalize(positive_outputs)
                # negative_outputs = l2_normalize(negative_outputs)
                positive_distance_list += p_dist.cpu().detach().numpy().tolist()
                negative_distance_list += n_dist.cpu().detach().numpy().tolist()
                if len(positive_distance_list) > 1000:
                    positive_distance_list = positive_distance_list[-1000:]
                if len(negative_distance_list) > 1000:
                    negative_distance_list = negative_distance_list[-1000:]
                # print("!!!!positive_distance_list:", sum(positive_distance_list) / len(positive_distance_list))
                # print("!!!!negative_distance_list:", sum(negative_distance_list) / len(negative_distance_list))

                eer = get_eer_for_two_distance_lists(
                    positive_distance_list, negative_distance_list)

                # Calculate loss
                # loss = modeltrainer.loss_function(l2_normalize(anchor_outputs), l2_normalize(positive_outputs), l2_normalize(negative_outputs))
                loss = modeltrainer.loss_function(
                    (anchor_outputs), (positive_outputs), (negative_outputs))
                running_wrong += modeltrainer.loss_function.wrong
                running_total += anchor_outputs.shape[0]
                loss.backward()

                if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader):
                    modeltrainer.optimizer.step()
                    modeltrainer.optimizer.zero_grad()
                    # torch.cuda.empty_cache()

                last_100_losses.append(loss.item())
                if len(last_100_losses) > 100:
                    last_100_losses.pop(0)
                running_loss += loss.item()

                progress_bar.set_postfix(loss=loss.item(),
                                         average_loss=sum(
                                             last_100_losses) / len(last_100_losses),
                                         wrongly_classified=f"{running_wrong}/{running_total} ({100*running_wrong/running_total:.5f}%)",
                                         eer=eer)

            except Exception as e:
                logger.error(
                    f"Error during training at step {step}: {e}", exc_info=True)
                torch.cuda.empty_cache()  # Clear cache in case of error
                continue

        logger.info(
            f"Completed epoch {epoch + 1}/{epochs} with running loss: {running_loss}.")
        return running_loss


class HardOfflineMiningTrainer(RandomMiningTrainer):
    def train_epoch(self, epoch, epochs, accumulation_steps, modeltrainer, create_dataset=None):
        logger.info(
            f"Starting hard offline mining for epoch {epoch + 1}/{epochs}.")
        with torch.no_grad():
            embeddings = []
            labels = []
            utterances = []
            running_loss = 0.0

            progress_bar = tqdm(
                modeltrainer.dataloader, desc=f"Epoch {epoch + 1}/{epochs}: Pre mining", leave=True)
            for step, (anchors, _, _, metadata) in enumerate(progress_bar):
                anchors = anchors.to(modeltrainer.device)
                anchor_outputs = modeltrainer.model(anchors)

                embeddings += [item for item in anchor_outputs]
                labels += [item["anchor_speaker"] for item in metadata]
                utterances += [item["anchor_utterance"] for item in metadata]

            triplets = hard_chunked_triplet_mining(
                embeddings, labels, modeltrainer.device)
            if len(triplets) == 0:
                logger.warning("No valid triplets found during mining.")
                return running_loss

            anchor_utterances = [utterances[i] for i in triplets[:, 0]]
            positive_utterances = [utterances[i] for i in triplets[:, 1]]
            negative_utterances = [utterances[i] for i in triplets[:, 2]]

            training_dataloader = create_dataset(
                anchor_utterances, positive_utterances, negative_utterances)
            torch.cuda.empty_cache()

        logger.info(
            f"Completed hard offline mining for epoch {epoch + 1}/{epochs}.")
        return self.train_epoch_triplets(epoch, epochs, accumulation_steps, training_dataloader, modeltrainer)


class DeepfakeHardOfflineMiningTrainer(RandomMiningTrainer):
    def train_epoch(self, epoch, epochs, accumulation_steps, modeltrainer, create_dataset=None, audio_dataset=None):
        logger.info(
            f"Starting deepfake hard offline mining for epoch {epoch + 1}/{epochs}.")
        with torch.no_grad():
            modeltrainer.model.eval()
            random_speaker = random.choice(
                audio_dataset.genuine['speaker'].unique())
            random_genuine_utterance = audio_dataset.genuine.sample(n=1)

            df = audio_dataset.data_list
            df = df[df["method_type"] != "Vocoder"]
            df = df[df["method_type"] != "bonafide"]
            df = df[df["is_genuine"] == 0]
            utterances_by_method = df[df['speaker'] == random_speaker].groupby(
                'method_name').apply(lambda x: x.sample(n=1))

            if random_genuine_utterance is not None:
                genuine_utterance_filename = random_genuine_utterance['filename'].values[0]
                genuine_utterance_data = audio_dataset.read_audio(
                    genuine_utterance_filename)
                genuine_utterance_embedding = modeltrainer.model(
                    genuine_utterance_data.unsqueeze(0).to(modeltrainer.device))

                distances = {}

                for index, row in utterances_by_method.iterrows():
                    method_name = row['method_name']
                    utterance_filename = row['filename']
                    utterance_data = audio_dataset.read_audio(
                        utterance_filename)

                    utterance_embedding = modeltrainer.model(
                        utterance_data.unsqueeze(0).to(modeltrainer.device))

                    distance = compute_distance(l2_normalize(
                        genuine_utterance_embedding), l2_normalize(utterance_embedding))
                    distances[method_name] = distance.item()

                best_method = min(distances, key=distances.get)
                logger.info(
                    f"Best method for speaker {random_speaker} is {best_method} with a distance of {distances[best_method]}.")
            else:
                logger.warning(
                    f"No genuine utterance found for speaker {random_speaker}.")

            torch.cuda.empty_cache()
            audio_dataset.set_method(best_method)
            modeltrainer.dataloader = create_dataset(audio_dataset)
            logger.info(
                f"Completed deepfake hard offline mining for epoch {epoch + 1}/{epochs}.")

        return self.train_epoch_triplets(epoch, epochs, accumulation_steps, modeltrainer.dataloader, modeltrainer)


class HardMiningTrainer(RandomMiningTrainer):
    def train_epoch(self, epoch, epochs, accumulation_steps, modeltrainer, create_dataset=None):
        logger.info(f"Starting hard mining for epoch {epoch + 1}/{epochs}.")
        modeltrainer.model.train()
        running_loss = 0.0
        last_100_losses = []
        progress_bar = tqdm(modeltrainer.dataloader,
                            desc=f"Epoch {epoch + 1}/{epochs}", leave=True)

        for step, (anchors, positives, negatives, metadata) in enumerate(progress_bar):
            anchors, positives, negatives = anchors.to(modeltrainer.device), positives.to(
                modeltrainer.device), negatives.to(modeltrainer.device)
            anchor_outputs = modeltrainer.model(anchors)
            positive_outputs = modeltrainer.model(positives)
            negative_outputs = modeltrainer.model(negatives)

            other_embeddings_torch = torch.cat(
                [anchor_outputs, positive_outputs, negative_outputs], dim=0)
            with torch.no_grad():
                embeddings = [item for item in anchor_outputs]
                labels = [item["anchor_speaker"] for item in metadata]
                other_embeddings = embeddings + \
                    [item for item in positive_outputs] + \
                    [item for item in negatives]
                other_labels = labels + [item["positive_speaker"] for item in metadata] + [
                    item["negative_speaker"] for item in metadata]
                triplets = hard_triplet_mining(
                    embeddings, labels, other_embeddings, other_labels, modeltrainer.device)

            if len(triplets) == 0:
                logger.warning(f"No valid triplets found at step {step}.")
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

        logger.info(
            f"Completed hard mining for epoch {epoch + 1}/{epochs} with running loss: {running_loss}.")
        return running_loss
