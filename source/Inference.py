import sys
import os
import warnings
import torch
from torch.nn import TripletMarginWithDistanceLoss

from dataloader import BSILoader, LibriSpeechLoader
from utils import get_device, get_inference_arguments, get_inference_variables, compute_distance
from models import WavLM_Base_ECAPA_TDNN, WavLM_Large_ECAPA_TDNN
from frontend import MFCCTransform
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
from source.utils.distance import  l2_normalize


def define_variables(args):
    global MODEL, DATASET, MFCCS, SAMPLE_RATE, EMBEDDING_SIZE, DEVICE, THRESHOLD, REFERENCE_AUDIO, QUESTION_AUDIO, QUESTION_AUDIO2

    MODEL, DATASET, MFCCS, SAMPLE_RATE, EMBEDDING_SIZE, DEVICE, THRESHOLD, REFERENCE_AUDIO, QUESTION_AUDIO, QUESTION_AUDIO2 = get_inference_variables(
        args)


def config():
    global device

    warnings.filterwarnings("ignore")

    device = get_device(DEVICE)


def create_dataset(args):
    global reference, question, question2

    if args.frontend == "mfcc":
        frontend = MFCCTransform(
            number_output_parameters=MFCCS, sample_rate=SAMPLE_RATE)
    else:
        def frontend(x): return x

    loader = DATASET.split(".")[0]
    if loader == "BSI":
        loader = BSILoader([], frontend, 0)
    elif loader == "LibriSpeech":
        loader = LibriSpeechLoader([], frontend, 0)
    # reference = loader.read_audio(REFERENCE_AUDIO, frontend).unsqueeze(0) TypeError: Loader.read_audio() takes 2 positional arguments but 3 were given
    # question = loader.read_audio(QUESTION_AUDIO, frontend).unsqueeze(0) TypeError: Loader.read_audio() takes 2 positional arguments but 3 were given

    reference = loader.read_audio(REFERENCE_AUDIO).unsqueeze(0)
    question = loader.read_audio(QUESTION_AUDIO).unsqueeze(0)
    if QUESTION_AUDIO2 is not None:
        question2 = loader.read_audio(QUESTION_AUDIO2).unsqueeze(0)
    else:
        question2 = None


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
    model.load_state_dict(state)
    model.eval()


def infer():

    triplet_loss = TripletMarginWithDistanceLoss(
        distance_function=compute_distance, margin=1)

    # reference_embedding = model(reference.to(device)).cpu().detach().numpy()
    # question_embedding = model(question.to(device)).cpu().detach().numpy()

    reference_embedding = model(reference.to(device))#.cpu().detach().numpy()
    question_embedding = model(question.to(device))#.cpu().detach().numpy()
    if question2 is not None:
        question2_embedding = model(question2.to(device))  # .cpu().detach().numpy()

    print("!!!!!!!!!!reference_embedding", reference_embedding)
    print("!!!!!!!!!!question_embedding", question_embedding)

    distance = compute_distance(reference_embedding, question_embedding)
    distance_normalized = compute_distance(l2_normalize(reference_embedding), l2_normalize(question_embedding))

    if question2 is not None:
        print("!!!!!!!!!!question2_embedding", question2_embedding)
        print("!!!!!!!!!!question2_embedding", question2_embedding)
        loss = triplet_loss(
            l2_normalize(reference_embedding), l2_normalize(question_embedding), l2_normalize(question2_embedding))
        print(f"!!!!!!!Loss: {loss}")

    print(f"Analyzing Audio {QUESTION_AUDIO} using model {MODEL}")
    print(f" > Reference audio: {REFERENCE_AUDIO}")
    print(f" > Distance between audio files: {distance}")
    print(f" > Normalized Distance between audio files: {distance_normalized}")
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
