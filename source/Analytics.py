import sys
import os
import warnings
from dataloader import ValidationDataset, collate_triplet_wav_fn, collate_valid_fn
from utils import load_deepfake_dataset, get_device, get_analytics_arguments, get_analytics_variables, ModelValidator
import torch
from torch.utils.data import DataLoader
from models import WavLM_Base_ECAPA_TDNN
from frontend import MFCCTransform
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
import pandas as pd


def define_variables(args):
    global MODEL, MFCCS, SAMPLE_RATE, EMBEDDING_SIZE, DOWNSAMPLING_TRAIN, DOWNSAMPLING_TEST, DOWNSAMPLING_VALID, BATCH_SIZE, TRAIN, VALID, TEST, NO_GENUINE, NO_DEEPFAKE

    MODEL, MFCCS, SAMPLE_RATE, EMBEDDING_SIZE, DOWNSAMPLING_TRAIN, DOWNSAMPLING_TEST, DOWNSAMPLING_VALID, BATCH_SIZE, TRAIN, VALID, TEST, NO_GENUINE, NO_DEEPFAKE = get_analytics_variables(
        args)


def config():
    global device

    warnings.filterwarnings("ignore")

    device = get_device()


def create_dataset(args):
    global audio_dataloader, validation_dataloader, test_dataloader

    train_labels, dev_labels, test_labels = load_deepfake_dataset()

    if args.frontend == "mfcc":
        frontend = MFCCTransform(
            number_output_parameters=MFCCS, sample_rate=SAMPLE_RATE)
    else:
        def frontend(x): return x

    if TRAIN:
        audio_dataset = ValidationDataset(
            train_labels, frontend=frontend, downsampling=DOWNSAMPLING_TRAIN)
        if NO_DEEPFAKE:
            audio_dataset.data_list = audio_dataset.data_list[
                audio_dataset.data_list["is_genuine"] == 1]
        audio_dataloader = DataLoader(audio_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                      drop_last=True, num_workers=4, pin_memory=True, collate_fn=collate_triplet_wav_fn)

    if VALID:
        validation_dataset = ValidationDataset(
            dev_labels, frontend=frontend, downsampling=DOWNSAMPLING_VALID)
        if NO_DEEPFAKE:
            validation_dataset.data_list = validation_dataset.data_list[
                validation_dataset.data_list["is_genuine"] == 1]
        validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                           drop_last=True, num_workers=4, pin_memory=True, collate_fn=collate_valid_fn)

    if TEST:
        test_dataset = ValidationDataset(
            test_labels, frontend=frontend, downsampling=DOWNSAMPLING_TEST)
        if NO_DEEPFAKE:
            test_dataset.data_list = test_dataset.data_list[test_dataset.data_list["is_genuine"] == 1]
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

    state = torch.load(
        f'../models/{MODEL}_best_model_state.pth')
    model.load_state_dict(state)
    model.eval()


def analyze():
    all_analytics = []

    if TRAIN:
        train_analytics = get_analytics("training dataset", audio_dataloader)
        print_analytics("Training Dataset", train_analytics)
        all_analytics.append(train_analytics)
    if VALID:
        valid_analytics = get_analytics(
            "validation dataset", validation_dataloader)
        print_analytics("Validation Dataset", valid_analytics)
        all_analytics.append(valid_analytics)
    if TEST:
        test_analytics = get_analytics("test dataset", test_dataloader)
        print_analytics("Test Dataset", test_analytics)
        all_analytics.append(test_analytics)

    if all_analytics:
        all_analytics_df = pd.concat(all_analytics, ignore_index=True)
        save_analytics_to_csv(all_analytics_df, "../data/analytics.csv")


def get_analytics(dataset_name, dataloader):
    print(
        f"Starting to validate the {dataset_name} (model={MODEL}, speaker_eer={(not NO_GENUINE)}, deepfake_eer={(not NO_DEEPFAKE)})")
    validator = ModelValidator(dataloader, device)
    sv_eer, sv_threshold, sv_rates, dd_eer, dd_threshold, dd_rates = validator.validate_model(
        model=model, speaker_eer=(not NO_GENUINE), deepfake_eer=(not NO_DEEPFAKE), mlflow_logging=False)

    data_single_row = {
        "Model": MODEL,
        "Dataset": dataset_name,
        "Speaker Verification EER": sv_eer,
        "Speaker Verification Threshold": sv_threshold,
        "Speaker Verification True Negatives": sv_rates["TN"],
        "Speaker Verification True Positives": sv_rates["TP"],
        "Speaker Verification False Negatives": sv_rates["FN"],
        "Speaker Verification False Positives": sv_rates["FP"],
        "Deepfake Detection EER": dd_eer,
        "Deepfake Detection Threshold": dd_threshold,
        "Deepfake Detection True Negatives": dd_rates["TN"],
        "Deepfake Detection True Positives": dd_rates["TP"],
        "Deepfake Detection False Negatives": dd_rates["FN"],
        "Deepfake Detection False Positives": dd_rates["FP"]
    }

    return pd.DataFrame([data_single_row])


def print_analytics(dataset_name, analytics_df):
    print(f"Results for {dataset_name} using model {MODEL}:\n")
    for _, row in analytics_df.iterrows():
        if not NO_GENUINE:
            print(
                f"Speaker Verification EER: {row['Speaker Verification EER']}")
            print(
                f"Speaker Verification Threshold: {row['Speaker Verification Threshold']}")
        if not NO_DEEPFAKE:
            print(f"Deepfake Detection EER: {row['Deepfake Detection EER']}")
            print(
                f"Deepfake Detection Threshold: {row['Deepfake Detection Threshold']}")


def save_analytics_to_csv(analytics_df, file_path):
    if os.path.exists(file_path):
        existing_df = pd.read_csv(file_path)
        analytics_df = pd.concat(
            [existing_df, analytics_df], ignore_index=True)

    analytics_df.to_csv(file_path, index=False)


def main(args):
    define_variables(args)

    ##### CONFIG #####
    config()

    ##### DATASET #####
    create_dataset(args)

    ##### MODEL DEFINITION #####
    get_model(args)

    ##### ANALYTICS #####
    analyze()


if __name__ == "__main__":
    args = get_analytics_arguments()

    os.chdir("./source")

    sys.exit(main(args))
