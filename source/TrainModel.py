import logging
import argparse
import sys
import os
import warnings
import mlflow
from utils import get_device, load_genuine_dataset, load_deepfake_dataset, ModelTrainer
import torch.optim as optim
from torch.nn import TripletMarginLoss
from torch.utils.data import DataLoader
from dataloader import ValidationDataset, RandomTripletLossDataset, DeepfakeRandomTripletLossDataset, collate_triplet_wav_fn, collate_valid_fn
from models import WavLM_Base_frozen_ECAPA_TDNN
from frontend import MFCCTransform
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


def define_variables(args):
    global MODEL, FOLDER, TAGS
    global LEARNING_RATE, MARGIN, NORM, BATCH_SIZE, EPOCHS, VALIDATION_RATE
    global MFCCS, SAMPLE_RATE, EMBEDDING_SIZE, WEIGHT_DECAY, AMSGRAD

    LEARNING_RATE = args.learning_rate
    MARGIN = args.margin
    NORM = args.norm
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    VALIDATION_RATE = args.validation_rate
    MFCCS = args.mfccs
    SAMPLE_RATE = args.sample_rate
    EMBEDDING_SIZE = args.embedding_size
    WEIGHT_DECAY = args.weight_decay
    AMSGRAD = args.amsgrad

    if args.frontend == "mfcc":
        MODEL = "MFCC"
        FOLDER = "MFCC"
        TAGS = {
            "Frontend": "MFCC"
        }
    else:
        if args.frontend == "wavlm_base":
            MODEL = "WavLM-Base"
            FOLDER = "WavLM-Base"
            TAGS = {
                "Frontend": "WavLM-Base"
            }
        elif args.frontend == "wavlm_large":
            MODEL = "WavLM-Large"
            FOLDER = "WavLM-Large"
            TAGS = {
                "Frontend": "WavLM-Large"
            }
        
        if args.frozen:
            MODEL += "-frozen"
            FOLDER += "/frozen"
            TAGS["Frontend-training"] = "frozen"
        else:
            MODEL += "-joint"
            FOLDER += "/joint"
            TAGS["Frontend-training"] = "joint"

    MODEL += "_ECAPA-TDNN_Random-Triplet-Mining"
    FOLDER += "/ECAPA-TDNN/Random-Triplet-Mining"
    TAGS["Model"] = "joint"
    TAGS["Triplet Mining Strategy"] = "Random Triplet Mining"
    
    if args.dataset == "genuine":
        MODEL += "_Genuine"
        FOLDER += "/Genuine"
        TAGS["Dataset"] = "Genuine"
    else:
        MODEL += "_Deepfake"
        FOLDER += "/Deepfake"
        TAGS["Dataset"] = "Deepfake"

def config():
    global logger
    global device

    warnings.filterwarnings("ignore")
    logging.basicConfig(filename=f'../logs/{MODEL}.log',
                        level=logging.INFO,
                        format='%(asctime)s - %(message)s')
    logger = logging.getLogger()

    mlflow.set_tracking_uri("../mlruns")
    logging.getLogger('mlflow.utils.requirements_utils').setLevel(logging.ERROR)

    device = get_device(logger)

def create_dataset(args):
    global audio_dataloader
    global validation_dataloader

    if args.dataset == "genuine":
        train_labels, dev_labels, test_labels = load_deepfake_dataset()
        tripletLossDataset = RandomTripletLossDataset
    elif args.dataset == "deepfake":
        train_labels, dev_labels, test_labels = load_deepfake_dataset()
        tripletLossDataset = DeepfakeRandomTripletLossDataset

    if args.frontend == "mfcc":
        frontend = MFCCTransform(number_output_parameters=MFCCS, sample_rate=SAMPLE_RATE)
    else:
        frontend = lambda x: x

    audio_dataset = tripletLossDataset(train_labels, frontend=frontend, logger=logger)
    audio_dataloader = DataLoader(audio_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4, pin_memory=True, collate_fn=collate_triplet_wav_fn)

    validation_dataset = ValidationDataset(dev_labels, frontend=frontend, logger=logger)
    validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4, pin_memory=True, collate_fn=collate_valid_fn)

def get_model(args):
    global model, optimizer, scheduler, triplet_loss

    if args.frontend == "mfcc":
        model = ECAPA_TDNN(input_size=MFCCS, lin_neurons=EMBEDDING_SIZE, device=device)
    elif args.frontend == "wavlm_base":
        if args.frozen:
            model = WavLM_Base_frozen_ECAPA_TDNN(device=device)
    
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, amsgrad=AMSGRAD)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,
        T_mult=1,
        eta_min=0.000005
    )
    triplet_loss = TripletMarginLoss(margin=MARGIN, p=NORM)


def main(args):
    define_variables(args)

    ##### CONFIG #####
    config()

    ##### DATASET #####
    create_dataset(args)

    ##### MODEL DEFINITION #####
    get_model(args)

    ##### TRAINING #####
    trainer = ModelTrainer(model, audio_dataloader, validation_dataloader, device, triplet_loss, optimizer, scheduler, logger, MODEL, validation_rate=VALIDATION_RATE, FOLDER=FOLDER, TAGS=TAGS)
    trainer.train_model(EPOCHS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training ECAPA-TDNN Model for Deepfake Speaker Verification")
    parser.add_argument(
        "--frontend",
        type=str,
        required=True,
        help="Which frontend model to use for feature extraction (mfcc, wavlm_base, wavlm_large)",
    )
    parser.add_argument(
        "--frozen",
        type=bool,
        required=False,
        default=True,
        help="Whether the frontend model is jointly trained or frozen during training",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Which dataset to use (genuine | deepfake)",
    )

    parser.add_argument("--learning_rate", type=float, required=False, default=0.001, help="")
    parser.add_argument("--weight_decay", type=float, required=False, default=0.00001, help="Weight decay to use for optimizing")
    parser.add_argument("--amsgrad", type=bool, required=False, default=False, help="Whether to use the AMSGrad variant of Adam")
    parser.add_argument("--margin", type=float, required=False, default=1, help="")
    parser.add_argument("--norm", type=int, required=False, default=2, help="")
    parser.add_argument("--batch_size", type=int, required=False, default=8, help="")
    parser.add_argument("--epochs", type=int, required=False, default=25, help="")
    parser.add_argument("--validation_rate", type=int, required=False, default=5, help="")
    parser.add_argument("--mfccs", type=int, required=False, default=13, help="")
    parser.add_argument("--sample_rate", type=int, required=False, default=16000, help="")
    parser.add_argument("--embedding_size", type=int, required=False, default=192, help="")

    args = parser.parse_args()

    os.chdir("./source")
    sys.exit(main(args))