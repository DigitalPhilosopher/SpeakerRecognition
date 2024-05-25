import logging
import sys
import os
import warnings
import mlflow
from utils import get_device, load_deepfake_dataset, ModelTrainer, get_training_arguments
import torch.optim as optim
from torch.nn import TripletMarginLoss
from torch.utils.data import DataLoader
from dataloader import ValidationDataset, RandomTripletLossDataset, DeepfakeRandomTripletLossDataset, collate_triplet_wav_fn, collate_valid_fn
from models import WavLM_Base_ECAPA_TDNN
from frontend import MFCCTransform
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


def define_variables(args):
    global MODEL, FOLDER, TAGS
    global LEARNING_RATE, MARGIN, NORM, BATCH_SIZE, EPOCHS, VALIDATION_RATE, RESTART_EPOCH
    global MFCCS, SAMPLE_RATE, EMBEDDING_SIZE, WEIGHT_DECAY, AMSGRAD
    global DOWNSAMPLING_TRAIN, DOWNSAMPLING_VALID, DOWNSAMPLING_TEST

    LEARNING_RATE = args.learning_rate
    RESTART_EPOCH = args.restart_epoch
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

    DOWNSAMPLING_TRAIN = args.downsample_train
    DOWNSAMPLING_TEST = args.downsample_test
    DOWNSAMPLING_VALID = args.downsample_valid

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
    global device

    warnings.filterwarnings("ignore")

    mlflow.set_tracking_uri("../mlruns")
    logging.getLogger(
        'mlflow.utils.requirements_utils').setLevel(logging.ERROR)

    device = get_device()


def create_dataset(args):
    global audio_dataloader, validation_dataloader, test_dataloader

    train_labels, dev_labels, test_labels = load_deepfake_dataset()

    if args.dataset == "genuine":
        tripletLossDataset = RandomTripletLossDataset
    elif args.dataset == "deepfake":
        tripletLossDataset = DeepfakeRandomTripletLossDataset

    if args.frontend == "mfcc":
        frontend = MFCCTransform(
            number_output_parameters=MFCCS, sample_rate=SAMPLE_RATE)
    else:
        def frontend(x): return x

    audio_dataset = tripletLossDataset(
        train_labels, frontend=frontend, downsampling=DOWNSAMPLING_TRAIN)
    audio_dataloader = DataLoader(audio_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  drop_last=True, num_workers=4, pin_memory=True, collate_fn=collate_triplet_wav_fn)

    validation_dataset = ValidationDataset(
        dev_labels, frontend=frontend, downsampling=DOWNSAMPLING_VALID)
    validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                       drop_last=True, num_workers=4, pin_memory=True, collate_fn=collate_valid_fn)

    test_dataset = ValidationDataset(
        test_labels, frontend=frontend, downsampling=DOWNSAMPLING_TEST)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                 drop_last=True, num_workers=4, pin_memory=True, collate_fn=collate_valid_fn)


def get_model(args):
    global model, optimizer, scheduler, triplet_loss

    if args.frontend == "mfcc":
        model = ECAPA_TDNN(
            input_size=MFCCS, lin_neurons=EMBEDDING_SIZE, device=device)
    elif args.frontend == "wavlm_base":
        if args.frozen == 0:
            frozen = False
        else:
            frozen = True
        model = WavLM_Base_ECAPA_TDNN(frozen=frozen, device=device)

    model.to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters(
    )), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, amsgrad=AMSGRAD)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=RESTART_EPOCH,
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
    trainer = ModelTrainer(model, audio_dataloader, validation_dataloader, test_dataloader, device, triplet_loss,
                           optimizer, scheduler, MODEL, validation_rate=VALIDATION_RATE, FOLDER=FOLDER, TAGS=TAGS)
    trainer.train_model(EPOCHS)


if __name__ == "__main__":
    args = get_training_arguments()

    os.chdir("./source")
    os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"

    sys.exit(main(args))
