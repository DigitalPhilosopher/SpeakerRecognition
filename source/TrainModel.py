import logging
import sys
import os
import warnings
import mlflow
from utils import get_device, load_deepfake_dataset, ModelTrainer, get_training_arguments, get_training_variables, compute_distance
import torch
import torch.optim as optim
from torch.nn import TripletMarginWithDistanceLoss
from torch.utils.data import DataLoader
from dataloader import ValidationDataset, RandomTripletLossDataset, HardTripletLossDataset, DeepfakeRandomTripletLossDataset, collate_triplet_wav_fn, collate_valid_fn, BSILoader, LibriSpeechLoader, VoxCelebLoader
from models import WavLM_Base_ECAPA_TDNN, WavLM_Large_ECAPA_TDNN
from frontend import MFCCTransform
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN


def define_variables(args):
    global MODEL, MODEL_PATH, DATASET, FOLDER, TAGS
    global LEARNING_RATE, MARGIN, NORM, BATCH_SIZE, BATCH_SIZE_TEST_EVAL, ACCUMULATION_STEPS, MAX_AUDIO_LENGTH, EPOCHS, VALIDATION_RATE
    global MFCCS, SAMPLE_RATE, EMBEDDING_SIZE, DEVICE, WEIGHT_DECAY, AMSGRAD
    global DOWNSAMPLING_TRAIN, DOWNSAMPLING_VALID, DOWNSAMPLING_TEST, TRIPLET_MINING

    (MODEL, MODEL_PATH, DATASET, FOLDER, TAGS, MFCCS, SAMPLE_RATE, EMBEDDING_SIZE,
     DEVICE, LEARNING_RATE, MARGIN, NORM, BATCH_SIZE, BATCH_SIZE_TEST_EVAL, ACCUMULATION_STEPS, MAX_AUDIO_LENGTH, EPOCHS,
     WEIGHT_DECAY, AMSGRAD, DOWNSAMPLING_TRAIN, DOWNSAMPLING_TEST,
     DOWNSAMPLING_VALID, TRIPLET_MINING) = get_training_variables(args)


def config():
    global device

    warnings.filterwarnings("ignore")

    mlflow.set_tracking_uri("../mlruns")
    logging.getLogger(
        'mlflow.utils.requirements_utils').setLevel(logging.ERROR)

    device = get_device(DEVICE)

def create_dataset_hard_mining(anchor, positive, negative):
    loader = DATASET.split(".")[0]
    if loader == "BSI":
        loader = BSILoader
    elif loader == "LibriSpeech":
        loader = LibriSpeechLoader
    elif loader == "VoxCeleb":
        loader = VoxCelebLoader
        
    if args.frontend == "mfcc":
        frontend = MFCCTransform(
            number_output_parameters=MFCCS, sample_rate=SAMPLE_RATE)
    else:
        def frontend(x): return x

    audio_dataset = HardTripletLossDataset(loader=loader(
        train_labels, frontend, DOWNSAMPLING_TRAIN), max_length=MAX_AUDIO_LENGTH)
    audio_dataset.set_triplets(anchor, positive, negative)
    audio_dataloader = DataLoader(audio_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                drop_last=True, num_workers=8, pin_memory=True, collate_fn=collate_triplet_wav_fn)

    return audio_dataloader


def create_dataset(args):
    global audio_dataloader, validation_dataloader, test_dataloader, train_labels

    train_labels, dev_labels, test_labels = load_deepfake_dataset(
        DATASET.split(".")[0])

    collate_triplet = collate_triplet_wav_fn
    data = DATASET.split(".")[1]
    if data == "genuine":
        tripletLossDataset = RandomTripletLossDataset
    elif data == "deepfake":
        tripletLossDataset = DeepfakeRandomTripletLossDataset

    if args.frontend == "mfcc":
        frontend = MFCCTransform(
            number_output_parameters=MFCCS, sample_rate=SAMPLE_RATE)
    else:
        def frontend(x): return x

    loader = DATASET.split(".")[0]
    if loader == "BSI":
        loader = BSILoader
    elif loader == "LibriSpeech":
        loader = LibriSpeechLoader
    elif loader == "VoxCeleb":
        loader = VoxCelebLoader

    audio_dataset = tripletLossDataset(loader=loader(
        train_labels, frontend, DOWNSAMPLING_TRAIN), max_length=MAX_AUDIO_LENGTH)
    audio_dataloader = DataLoader(audio_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  drop_last=True, num_workers=8, pin_memory=True, collate_fn=collate_triplet)

    validation_dataloader = None
    if DOWNSAMPLING_VALID > 0:
        validation_dataset = ValidationDataset(
            loader=loader(dev_labels, frontend, DOWNSAMPLING_VALID), max_length=MAX_AUDIO_LENGTH)
        validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE_TEST_EVAL, shuffle=True,
                                           drop_last=True, num_workers=4, pin_memory=True, collate_fn=collate_valid_fn)

    test_dataloader = None
    if DOWNSAMPLING_TEST > 0:
        test_dataset = ValidationDataset(loader=loader(
            test_labels, frontend, DOWNSAMPLING_TEST), max_length=MAX_AUDIO_LENGTH)
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE_TEST_EVAL, shuffle=True,
                                     drop_last=True, num_workers=4, pin_memory=True, collate_fn=collate_valid_fn)


def get_model(args):
    global model, optimizer, triplet_loss

    if args.frozen == 0:
        frozen = False
    else:
        frozen = True

    if args.frontend == "mfcc":
        model = ECAPA_TDNN(
            input_size=MFCCS, lin_neurons=EMBEDDING_SIZE, device=device)
    elif args.frontend == "wavlm_base":
        model = WavLM_Base_ECAPA_TDNN(frozen=frozen, device=device)

    elif args.frontend == "wavlm_large":
        model = WavLM_Large_ECAPA_TDNN(frozen=frozen, device=device)
    # Load pretrained model if MODEL_PATH is provided
    if MODEL_PATH is not None:
        print(f"Loading pretrained model from {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH))

    model.to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters(
    )), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, amsgrad=AMSGRAD)
    triplet_loss = TripletMarginWithDistanceLoss(
        distance_function=compute_distance, margin=MARGIN)


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
                           optimizer, MODEL, FOLDER=FOLDER, TAGS=TAGS, accumulation_steps=ACCUMULATION_STEPS)
    trainer.train_model(EPOCHS, triplet_mining=TRIPLET_MINING, create_dataset = lambda a, p, n: create_dataset_hard_mining(a, p, n))


if __name__ == "__main__":
    args = get_training_arguments()

    os.chdir("./source")
    os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"

    sys.exit(main(args))
