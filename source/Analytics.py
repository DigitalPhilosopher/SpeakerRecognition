import sys
import os
import warnings
from dataloader import ValidationDataset, collate_valid_fn, BSILoader, LibriSpeechLoader, VoxCelebLoader
from utils import load_deepfake_dataset, get_device, get_analytics_arguments, get_analytics_variables, ModelValidator, get_valid_sets, get_train_sets, get_test_sets
import torch
from torch.utils.data import DataLoader
from models import WavLM_Base_ECAPA_TDNN, WavLM_Large_ECAPA_TDNN
from frontend import MFCCTransform
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
import pandas as pd


def define_variables(args):
    global MODEL, DATASET, MFCCS, SAMPLE_RATE, EMBEDDING_SIZE, DEVICE, DOWNSAMPLING_TRAIN, DOWNSAMPLING_TEST, DOWNSAMPLING_VALID, MAX_AUDIOS_TRAIN, MAX_AUDIOS_TEST, MAX_AUDIOS_VALID, BATCH_SIZE, TRAIN, VALID, TEST, NO_GENUINE, NO_DEEPFAKE, NO_VALID_SET

    MODEL, DATASET, MFCCS, SAMPLE_RATE, EMBEDDING_SIZE, DEVICE, DOWNSAMPLING_TRAIN, DOWNSAMPLING_TEST, DOWNSAMPLING_VALID, MAX_AUDIOS_TRAIN, MAX_AUDIOS_TEST, MAX_AUDIOS_VALID, BATCH_SIZE, TRAIN, VALID, TEST, NO_GENUINE, NO_DEEPFAKE, NO_VALID_SET = get_analytics_variables(
        args)


def config():
    global device

    warnings.filterwarnings("ignore")

    device = get_device(DEVICE)


def create_dataset(args):
    global audio_dataloader, validation_dataloader, test_dataloader

    train_labels, dev_labels, test_labels = load_deepfake_dataset(
        DATASET.split(".")[0])


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

    if TRAIN:
        audio_dataset = ValidationDataset(loader=loader(
            train_labels, frontend, DOWNSAMPLING_TRAIN, MAX_AUDIOS_TRAIN))
        if NO_DEEPFAKE:
            audio_dataset.data_list = audio_dataset.data_list[
                audio_dataset.data_list["is_genuine"] == 1]
        audio_dataloader = DataLoader(audio_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                      drop_last=False, num_workers=4, pin_memory=True, collate_fn=collate_valid_fn)

    if VALID:
        validation_dataset = ValidationDataset(
            loader=loader(dev_labels, frontend, DOWNSAMPLING_VALID, MAX_AUDIOS_VALID))
        if NO_DEEPFAKE:
            validation_dataset.data_list = validation_dataset.data_list[
                validation_dataset.data_list["is_genuine"] == 1]
        validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                           drop_last=False, num_workers=4, pin_memory=True, collate_fn=collate_valid_fn)

    if TEST:
        print("!!!!!!!MAX_AUDIOS_TEST:", MAX_AUDIOS_TEST)
        test_dataset = ValidationDataset(loader=loader(
            test_labels, frontend, DOWNSAMPLING_TEST, MAX_AUDIOS_TEST))
        if NO_DEEPFAKE:
            test_dataset.data_list = test_dataset.data_list[test_dataset.data_list["is_genuine"] == 1]
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                     drop_last=False, num_workers=4, pin_memory=True, collate_fn=collate_valid_fn)


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

    model.to(device)

    state = torch.load(
        f'../models/{MODEL}_best_model_state.pth')
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()


def analyze():
    all_analytics = []

    if TRAIN:
        _set, df_set = get_train_sets(DATASET.split(".")[0])
        if NO_VALID_SET:
            _set = []
        train_analytics = get_analytics(
            "training dataset", audio_dataloader, _set, df_set)
        print_analytics("Training Dataset", train_analytics)
        all_analytics.append(train_analytics)
    if VALID:
        _set, df_set = get_valid_sets(DATASET.split(".")[0])
        if NO_VALID_SET:
            _set = []
        valid_analytics = get_analytics(
            "validation dataset", validation_dataloader, _set, df_set)
        print_analytics("Validation Dataset", valid_analytics)
        all_analytics.append(valid_analytics)
    if TEST:
        _set, df_set = get_test_sets(DATASET.split(".")[0])
        if NO_VALID_SET:
            _set = []
        test_analytics = get_analytics(
            "test dataset", test_dataloader, _set, df_set)
        print_analytics("Test Dataset", test_analytics)
        all_analytics.append(test_analytics)

    if all_analytics:
        all_analytics_df = pd.concat(all_analytics, ignore_index=True)
        save_analytics_to_csv(all_analytics_df, "../data/analytics.csv")


def get_analytics(dataset_name, dataloader, _set, df_set):
    print(
        f"Starting to validate the {dataset_name} (model={MODEL}, speaker_eer={(not NO_GENUINE)}, deepfake_eer={(not NO_DEEPFAKE)})")
    validator = ModelValidator(dataloader, device, _set, df_set)
    sv_eer, sv_threshold, sv_rates, sv_minDCF, dd_eer, dd_threshold, dd_rates, dd_minDCF = validator.validate_model(
        model=model, speaker_eer=(not NO_GENUINE), deepfake_eer=(not NO_DEEPFAKE), mlflow_logging=False)

    data_single_row = {
        "Model": MODEL,
        "Dataset": dataset_name,
        "Speaker Verification EER": sv_eer,
        "Speaker Verification minDCF": sv_minDCF,
        "Speaker Verification Threshold": sv_threshold,
        "Speaker Verification True Negatives": sv_rates["TN"],
        "Speaker Verification True Positives": sv_rates["TP"],
        "Speaker Verification False Negatives": sv_rates["FN"],
        "Speaker Verification False Positives": sv_rates["FP"],
        "Deepfake Detection EER": dd_eer,
        "Deepfake Detection minDCF": dd_minDCF,
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
                f"Speaker Verification minDCF: {row['Speaker Verification minDCF']}")
            print(
                f"Speaker Verification Threshold: {row['Speaker Verification Threshold']}")
        if not NO_DEEPFAKE:
            print(f"Deepfake Detection EER: {row['Deepfake Detection EER']}")
            print(
                f"Deepfake Detection minDCF: {row['Deepfake Detection minDCF']}")
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
