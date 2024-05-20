import logging
import mlflow
from utils import get_device, load_genuine_dataset, ModelTrainer
import torch.optim as optim
from torch.nn import TripletMarginLoss
from torch.utils.data import DataLoader
from dataloader import ValidationDataset, RandomTripletLossDataset, collate_triplet_wav_fn, collate_valid_fn
from models import WavLM_Base_frozen_ECAPA_TDNN


##### VARIABLES #####
MODEL = "WavLM-Base-frozen_ECAPA-TDNN_Genuine_Random"
FOLDER = "WavLM-Base/ECAPA-TDNN/frozen/genuine"
TAGS = {
    "Frontend": "WavLM-Base",
    "Model": "ECAPA-TDNN",
    "Frontend-training" : "frozen",
    "Dataset" : "genuine",
    "Triplet Mining Strategy" : "Random Triplet Mining"
}

LEARNING_RATE = 0.001
MARGIN = 1
NORM = 2

BATCH_SIZE = 8
EPOCHS = 4
VALIDATION_RATE = 1


def main():
    ##### CONFIG #####
    logging.basicConfig(filename=f'../logs/{MODEL}.log',
                        level=logging.INFO,
                        format='%(asctime)s - %(message)s')
    logger = logging.getLogger()

    mlflow.set_tracking_uri("../mlruns")
    logging.getLogger('mlflow.utils.requirements_utils').setLevel(logging.ERROR)

    device = get_device(logger)

    ##### DATASET #####
    train_labels, dev_labels, test_labels = load_genuine_dataset()

    audio_dataset = RandomTripletLossDataset(train_labels, frontend=lambda x: x, logger=logger)
    audio_dataloader = DataLoader(audio_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_triplet_wav_fn)

    validation_dataset = ValidationDataset(dev_labels, frontend=lambda x: x, logger=logger)
    validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_valid_fn)


    ##### MODEL DEFINITION #####
    model = WavLM_Base_frozen_ECAPA_TDNN(device=device)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    triplet_loss = TripletMarginLoss(margin=MARGIN, p=NORM)

    ##### TRAINING #####
    trainer = ModelTrainer(model, audio_dataloader, validation_dataloader, device, triplet_loss, optimizer, logger, MODEL, validation_rate=VALIDATION_RATE, FOLDER=FOLDER, TAGS=TAGS)
    trainer.train_model(EPOCHS)


if __name__ == '__main__':
    main()