import sys
import os
import warnings
from dataloader import read_audio
from utils import get_device, get_inference_arguments, get_inference_variables, compute_distance
import torch
from models import WavLM_Base_ECAPA_TDNN
from frontend import MFCCTransform
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN


def define_variables(args):
    global MODEL, MFCCS, SAMPLE_RATE, EMBEDDING_SIZE, DEVICE, THRESHOLD, REFERENCE_AUDIO, QUESTION_AUDIO

    MODEL, MFCCS, SAMPLE_RATE, EMBEDDING_SIZE, DEVICE, THRESHOLD, REFERENCE_AUDIO, QUESTION_AUDIO = get_inference_variables(
        args)


def config():
    global device

    warnings.filterwarnings("ignore")

    device = get_device(DEVICE)


def create_dataset(args):
    global reference, question

    if args.frontend == "mfcc":
        frontend = MFCCTransform(
            number_output_parameters=MFCCS, sample_rate=SAMPLE_RATE)
    else:
        def frontend(x): return x

    reference = read_audio(REFERENCE_AUDIO, frontend).unsqueeze(0)
    question = read_audio(QUESTION_AUDIO, frontend).unsqueeze(0)


def get_model(args):
    global model, optimizer, triplet_loss

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


def infer():
    reference_embedding = model(reference.to(device)).cpu().detach().numpy()
    question_embedding = model(question.to(device)).cpu().detach().numpy()

    distance = compute_distance(reference_embedding, question_embedding)

    print(f"Analyzing Audio {QUESTION_AUDIO} using model {MODEL}")
    print(f" > Reference audio: {REFERENCE_AUDIO}")
    print(f" > Distance between audio files: {distance}")
    if distance > THRESHOLD:
        print(f"\n => DECLINE: NOT THE SAME SPEAKER")
    else:
        print(f"\n => ACCEPT: SAME SPEAKER")


def main(args):
    define_variables(args)

    ##### CONFIG #####
    config()

    ##### DATASET #####
    create_dataset(args)

    ##### MODEL DEFINITION #####
    get_model(args)

    ##### INFERENCE #####
    infer()


if __name__ == "__main__":
    args = get_inference_arguments()

    os.chdir("./source")
    os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"

    sys.exit(main(args))
