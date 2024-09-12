# -------------------------------------------------- #
# -------------------------------------------------- #
# --------------------Arguments--------------------- #
# -------------------------------------------------- #
# -------------------------------------------------- #

import argparse
import os

parser = argparse.ArgumentParser(description='Model Setup Parameters')
parser.add_argument('--dir', type=str, required=True, help='Model directory', default="models")
parser.add_argument('--model', type=str, required=True, help='Model checkpoint file', default="WavLM-Base-joint_ECAPA-TDNN_Random-Triplet-Mining_BSI-Deepfake_checkpoint.pth")
parser.add_argument('--batches', type=int, default=4, help='Number of batches')
parser.add_argument('--labels', type=str, default='valid', help='Labels (train/test/valid)')
parser.add_argument('--num_speakers', type=int, default=10, help='Number of speakers')
parser.add_argument('--num_deepfakes', type=int, default=10, help='Number of deepfakes')
parser.add_argument('--distances', type=int, default=10, help='Distance value')
parser.add_argument('--max_length', type=int, default=32000, help='Maximum length')
parser.add_argument('--dataset', type=str, default='BSI.deepfake', help='Dataset name')

args = parser.parse_args()

DIR = args.dir
MODEL = args.model
BATCHES = args.batches
LABELS = args.labels
NUMBER_OF_SPEAKER = args.num_speakers
NUMBER_OF_DEEPFAKES = args.num_deepfakes
DISTANCES = args.distances
MAX_LENGTH = args.max_length
DATASET = args.dataset

MODEL_PATH = f'{DIR}{MODEL}'
SAVE = f"{DIR}analytics/{DATASET}/{LABELS}/{MODEL.split('_')[0]}"
os.makedirs(SAVE, exist_ok=True)
for i in range(1, 7):
    os.makedirs(os.path.join(SAVE, f"check_{i}"), exist_ok=True)

# -------------------------------------------------- #
# -------------------------------------------------- #
# --------------------Imports----------------------- #
# -------------------------------------------------- #
# -------------------------------------------------- #

import torch
from models import WavLM_Base_ECAPA_TDNN, WavLM_Large_ECAPA_TDNN
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
from frontend import MFCCTransform
import pandas as pd
import librosa
import math
import random
import itertools
from utils.distance import compute_distance
from sklearn.metrics import roc_curve
import numpy as np
from tqdm import tqdm
import plotly.express as px
from utils import load_deepfake_dataset
from dataloader import RandomTripletLossDataset, BSILoader
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
import warnings
import logging

logging.getLogger('s3prl.util.download').setLevel(logging.ERROR)
logging.getLogger('s3prl.upstream.wavlm.WavLM').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*has been deprecated.*")
logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore")

# -------------------------------------------------- #
# -------------------------------------------------- #
# --------------------Functions--------------------- #
# -------------------------------------------------- #
# -------------------------------------------------- #

def read_audio_batch(filenames, frontend, max_length=0):
    waveforms = []
    for filename in filenames:
        waveform, _ = librosa.load(filename, sr=16000)  # Load audio file
        waveform, _ = librosa.effects.trim(waveform, top_db=35)  # Trim silence
        waveform = torch.tensor(waveform, dtype=torch.float32)  # Convert to torch tensor
        if max_length > 0 and waveform.shape[0] > max_length:
            # Randomly select a segment if waveform is longer than max_length
            start_sample = random.randint(0, waveform.shape[-1] - max_length)
            end_sample = start_sample + max_length
            waveform = waveform[start_sample:end_sample]
        waveforms.append(frontend(waveform))
    
    # Pad waveforms to the same length
    waveforms = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
    
    return waveforms

def extract_embeddings_batch(model, waveforms):
    waveforms = waveforms.to('cuda')  # Add channel dimension and move to GPU
    with torch.no_grad():
        embeddings = model(waveforms)  # Get embeddings from the model
    #return l2_normalize(embeddings.cpu())  # Normalize and move back to CPU
    return embeddings.cpu()  # Normalize and move back to CPU

def add_check_distances(check_data):
    speakers = check_data["speaker"].unique()
    for speaker in tqdm(speakers, desc="Speakers", total=len(speakers), position=0):
        speaker_data = check_data[check_data["speaker"] == speaker]
        for idx, deepfake in tqdm(speaker_data.iterrows(), desc="Deepfakes", position=1, leave=False):
            unique_ids = list(set([item for sublist in deepfake["positive_combinations"] for item in sublist]))
            number_of_stacks = len(unique_ids)
            if len(unique_ids) > DISTANCES:
                number_of_stacks = DISTANCES
                unique_ids = unique_ids[:DISTANCES]

            embeddings = []
            for utterance in unique_ids:
                embedding = data_list.loc[(data_list['utterance'] == utterance) & (data_list['is_genuine'] == 1), 'embeddings'].values[0]
                embeddings.append(torch.tensor(embedding))
            embeddings = torch.stack(embeddings)

            stacked_embeddings = torch.stack([torch.tensor(deepfake["embeddings"])] * number_of_stacks)

            distances = compute_distance(embeddings, stacked_embeddings)
            data_list.at[idx, "check_distances"] = distances.tolist()
            data_list.at[idx, "check_utterances"] = unique_ids

def add_check(check, data_list):
    # Filter out rows where 'check' or 'is_genuine' are NaN
    valid_data = data_list[[check, 'is_genuine']].dropna(subset=[check, 'is_genuine'])
    
    # Check if there are any valid values left
    if valid_data.empty:
        data_list[f"{check}_is_genuine"] = np.nan
        data_list[f"{check}_eer"] = np.nan
        data_list[f"{check}_threshold"] = np.nan

        return data_list

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(1 - valid_data['is_genuine'], valid_data[check])
    
    # Calculate the false negative rate (FNR) and equal error rate (EER)
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.absolute(fpr - fnr))
    eer = fpr[eer_index]
    eer_threshold = thresholds[eer_index]

    # Add new columns based on the EER threshold
    new_columns = pd.DataFrame({
        f"{check}_is_genuine": data_list[check].apply(lambda x: x < eer_threshold if not np.isnan(x) else np.nan),
        f"{check}_eer": eer,
        f"{check}_threshold": eer_threshold
    })

    # Concatenate the new columns to the original DataFrame
    data_list = pd.concat([data_list, new_columns], axis=1)

    return data_list

def add_all_checks(data_list):
    data_list = add_check("check_1", data_list)
    data_list = add_check("check_2", data_list)
    for i in range(DISTANCES):
        data_list = add_check(f"check_3_{i+1}", data_list)
        data_list = add_check(f"check_4_{i+1}", data_list)
        data_list = add_check(f"check_5_{i+1}", data_list)
        data_list = add_check(f"check_6_{i+1}", data_list)
    return data_list

def create_bar_figure(check, data_list, with_vocoder=False):
    method_names = []
    correct = []
    wrong = []

    total_true   = len(data_list[(data_list[f"{check}_is_genuine"] == True)  & (data_list["is_genuine"] == 1)])
    total_true   += len(data_list[(data_list[f"{check}_is_genuine"] == False)  & (data_list["is_genuine"] == 0)])
    total_false  = len(data_list[(data_list[f"{check}_is_genuine"] == False) & (data_list["is_genuine"] == 1) ])
    total_false  += len(data_list[(data_list[f"{check}_is_genuine"] == True) & (data_list["is_genuine"] == 0) ])
    method_names.append("Total")
    correct.append(total_true)
    wrong.append(total_false)

    if with_vocoder:
        total_ex_vocoder_true   = len(data_list[(data_list[f"{check}_is_genuine"] == True)  & (data_list["is_genuine"] == 1) & (data_list["method_type"] != "Vocoder")])
        total_ex_vocoder_true   += len(data_list[(data_list[f"{check}_is_genuine"] == False)  & (data_list["is_genuine"] == 0) & (data_list["method_type"] != "Vocoder")])
        total_ex_vocoder_false  = len(data_list[(data_list[f"{check}_is_genuine"] == False) & (data_list["is_genuine"] == 1)  & (data_list["method_type"] != "Vocoder")])
        total_ex_vocoder_false  += len(data_list[(data_list[f"{check}_is_genuine"] == True) & (data_list["is_genuine"] == 0)  & (data_list["method_type"] != "Vocoder")])
        method_names.append("Total/Vocoder")
        correct.append(total_ex_vocoder_true)
        wrong.append(total_ex_vocoder_false)

    bonafide_true   = len(data_list[(data_list[f"{check}_is_genuine"] == True)  & (data_list["is_genuine"] == 1) & (data_list["method_type"] != "Vocoder")])
    bonafide_false  = len(data_list[(data_list[f"{check}_is_genuine"] == False) & (data_list["is_genuine"] == 1) & (data_list["method_type"] != "Vocoder")])
    method_names.append("Bonafide")
    correct.append(bonafide_true)
    wrong.append(bonafide_false)

    if with_vocoder:
        vocoder_true   = len(data_list[(data_list[f"{check}_is_genuine"] == True)  & (data_list["is_genuine"] == 1) & (data_list["method_type"] == "Vocoder")])
        vocoder_false  = len(data_list[(data_list[f"{check}_is_genuine"] == False) & (data_list["is_genuine"] == 1) & (data_list["method_type"] == "Vocoder")])
        method_names.append("Vocoder")
        correct.append(vocoder_true)
        wrong.append(vocoder_false)

    deepfake_true   = len(data_list[(data_list[f"{check}_is_genuine"] == False)  & (data_list["is_genuine"] == 0)])
    deepfake_false  = len(data_list[(data_list[f"{check}_is_genuine"] == True) & (data_list["is_genuine"] == 0)])
    method_names.append("Deepfake")
    correct.append(deepfake_true)
    wrong.append(deepfake_false)

    tts_true   = len(data_list[(data_list[f"{check}_is_genuine"] == False)  & (data_list["method_type"] == "TTS")])
    tts_false  = len(data_list[(data_list[f"{check}_is_genuine"] == True) & (data_list["method_type"] == "TTS")])
    method_names.append("TTS")
    correct.append(tts_true)
    wrong.append(tts_false)

    vc_true   = len(data_list[(data_list[f"{check}_is_genuine"] == False)  & (data_list["method_type"] == "VC")])
    vc_false  = len(data_list[(data_list[f"{check}_is_genuine"] == True) & (data_list["method_type"] == "VC")])
    method_names.append("VC")
    correct.append(vc_true)
    wrong.append(vc_false)


    fig = go.Figure(data=[
        go.Bar(name='Correct classified', x=method_names, y=correct),
        go.Bar(name='Wrong classified'  , x=method_names, y=wrong)
    ])

    fig.update_layout(barmode='group')
    return fig, method_names, correct, wrong

def fig_confusion(check, data_list):
    # Example true labels and predicted labels
    true_labels = data_list["is_genuine"]
    predicted_labels = data_list[f"{check}_is_genuine"].apply(lambda x: 1 if x else 0)

    # Compute confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
                    z=conf_matrix,
                    x=['Predicted Negative', 'Predicted Positive'],
                    y=['Actual Negative', 'Actual Positive'],
                    hoverongaps=False,
                    colorscale='Greens'))

    # Add annotations (optional)
    annotations = []
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            annotations.append(
                dict(
                    x=j,
                    y=i,
                    text=str(conf_matrix[i][j]),
                    showarrow=False,
                    font=dict(color="black")
                )
            )

    fig.update_layout(
        title='Confusion Matrix',
        annotations=annotations,
        xaxis_title='Predicted label',
        yaxis_title='True label'
    )
    return fig

def create_check_figures(check):
    fig, method_names, correct, wrong = create_bar_figure(check, data_list)
    fig.write_image(f"{SAVE}/barplot.png")
    for i in range(len(method_names)):
        nclass.append([
            check,
            False,
            method_names[i],
            correct[i],
            wrong[i]
        ])

    fig, method_names, correct, wrong = create_bar_figure(check, data_list_with_vocoder, True)
    fig.write_image(f"{SAVE}/barplot_with_vocoder.png")
    for i in range(len(method_names)):
        nclass.append([
            check,
            True,
            method_names[i],
            correct[i],
            wrong[i]
        ])

    fig = fig_confusion(check, data_list)
    fig.write_image(f"{SAVE}/confusion_matrix.png")

def create_viz(check):
    viz = []
    for i in range(DISTANCES):
        if pd.isna(data_list[f"{check}_{i+1}_eer"].iloc[0]):
            continue
        viz.append([
            i+1,
            data_list[f"{check}_{i+1}_eer"].iloc[0],
            "TOTAL"
        ])
        viz.append([
            i+1,
            data_list_with_vocoder[f"{check}_{i+1}_eer"].iloc[0],
            "TOTAL with Vocoder"
        ])
        viz.append([
            i+1,
            tts_data[f"{check}_{i+1}_eer"].iloc[0],
            "TTS"
        ])
        viz.append([
            i+1,
            vc_data[f"{check}_{i+1}_eer"].iloc[0],
            "Voice Conversion"
        ])

    column_names = ['Number of checks', 'EER', 'Method Type']
    df = pd.DataFrame(viz, columns=column_names)

    fig = px.line(df, x='Number of checks', y='EER', color='Method Type', markers=True)
    fig.write_image(f"{SAVE}/{check}_eer_per_distance.png")

# -------------------------------------------------- #
# -------------------------------------------------- #
# --------------------Model and Data---------------- #
# -------------------------------------------------- #
# -------------------------------------------------- #

if MODEL.split("_")[0].lower().startswith("mfcc"):
    model = ECAPA_TDNN(input_size=80, lin_neurons=192)
elif MODEL.split("_")[0].lower().startswith("wavlm"):
    if MODEL.split("-")[1].lower() == "base":
        model = WavLM_Base_ECAPA_TDNN(frozen=True)
    else:
        model = WavLM_Large_ECAPA_TDNN(frozen=True)
try:
    model.load_state_dict(torch.load(MODEL_PATH))
except:
    checkpoint = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint['state_dict'])
model.eval()
model.to('cuda')

train_labels, valid_labels, test_labels = load_deepfake_dataset("BSI")
if LABELS == "train":
    labels = train_labels
elif LABELS == "test":
    labels = test_labels
else:
    labels = valid_labels


if MODEL.split("_")[0].lower().startswith("mfcc"):
    frontend = MFCCTransform(
        number_output_parameters=80, sample_rate=16000)
else:
    def frontend(x): return x

dataset = RandomTripletLossDataset(loader=BSILoader(labels, frontend, 0))
data_list = dataset.data_list
speaker_counts = dataset.genuine['speaker'].value_counts()
data_list = data_list[data_list['speaker'].map(speaker_counts) >= 2].reset_index(drop=True)

if NUMBER_OF_SPEAKER > 0:
    speakers = data_list['speaker'].unique()
    selected_speakers = random.sample(list(speakers), NUMBER_OF_SPEAKER)
    data_list = data_list[data_list['speaker'].isin(selected_speakers)]

if NUMBER_OF_DEEPFAKES > 0:
    grouped = data_list.groupby(['method_name', 'speaker'])
    sampled_deepfakes = []

    for _, group in grouped:
        deepfakes = group[group['is_genuine'] == 0]
        if len(deepfakes) >= NUMBER_OF_DEEPFAKES:
            sampled = deepfakes.sample(NUMBER_OF_DEEPFAKES)
            sampled_deepfakes.append(sampled)
        else:
            sampled_deepfakes.append(deepfakes)

    sampled_deepfakes_df = pd.concat(sampled_deepfakes)
    genuine_entries = data_list[data_list["is_genuine"] == 1]
    data_list = pd.concat([genuine_entries, sampled_deepfakes_df])


# -------------------------------------------------- #
# -------------------------------------------------- #
# --------------------Embeddings-------------------- #
# -------------------------------------------------- #
# -------------------------------------------------- #

def chunker(data, chunk_size):
    for start in range(0, len(data), chunk_size):
        yield data.iloc[start:start + chunk_size]

total_chunks = len(data_list) // BATCHES + (1 if len(data_list) % BATCHES != 0 else 0)
all_embeddings = []
for chunk in tqdm(chunker(data_list, BATCHES), total=total_chunks):
    waveforms = read_audio_batch(chunk['filename'].tolist(), frontend, MAX_LENGTH)

    embeddings = extract_embeddings_batch(model, waveforms)
    embeddings = embeddings.squeeze(1)  # Shape becomes [batch_size, embedding_dim]

    all_embeddings.append(embeddings.cpu().numpy())

all_embeddings = np.vstack(all_embeddings)
data_list['embeddings'] = [emb for emb in all_embeddings]

# -------------------------------------------------- #
# -------------------------------------------------- #
# --------------------Positive combinations--------- #
# -------------------------------------------------- #
# -------------------------------------------------- #

data_list["positive_distances"] = None
data_list["positive_combinations"] = None
data_list["check_distances"] = None
data_list["check_utterances"] = None

speakers = data_list["speaker"].unique()
genuine_data = data_list[data_list['is_genuine'] == 1]

for speaker in tqdm(speakers, desc="Speakers", total=len(speakers), position=0):
    same_speakers = genuine_data[genuine_data["speaker"] == speaker]
    MAX_DISTANCES = min(math.comb(len(same_speakers), 2), DISTANCES)
    
    all_combinations = list(itertools.combinations(same_speakers.index, 2))
    selected_combinations = all_combinations[:MAX_DISTANCES]
    if len(selected_combinations) < 1:
        continue

    batch1 = []
    batch2 = []
    combinations = []
    for pair in selected_combinations:
        idx1, idx2 = pair
        embedding1 = same_speakers.loc[idx1, 'embeddings']
        embedding2 = same_speakers.loc[idx2, 'embeddings']
        utterance1 = same_speakers.loc[idx1, 'utterance']
        utterance2 = same_speakers.loc[idx2, 'utterance']
        batch1.append(torch.tensor(embedding1))
        batch2.append(torch.tensor(embedding2))
        combinations.append([utterance1, utterance2])
    
    batch1 = torch.stack(batch1)
    batch2 = torch.stack(batch2)
    
    calculated_distances = compute_distance(batch1, batch2).tolist()
    data_list.loc[data_list["speaker"] == speaker, "positive_distances"] = data_list.loc[data_list["speaker"] == speaker, "positive_distances"].apply(
        lambda x: (x if x is not None else []) + calculated_distances
    )
    data_list.loc[data_list["speaker"] == speaker, "positive_combinations"] = data_list.loc[data_list["speaker"] == speaker, "positive_combinations"].apply(
        lambda x: (x if x is not None else []) + combinations
    )

# -------------------------------------------------- #
# -------------------------------------------------- #
# --------------------Check combinations------------ #
# -------------------------------------------------- #
# -------------------------------------------------- #

data_list = data_list[~pd.isna(data_list["positive_distances"])]
data_list["is_genuine"] = data_list["method_type"].apply(lambda x: 1 if (x == "Vocoder") | (x == "bonafide") else 0)

bonafide_data = data_list[data_list['method_type'] == "bonafide"]
add_check_distances(bonafide_data)

vocoder_data = data_list[data_list['method_type'] == "Vocoder"]
add_check_distances(vocoder_data)

tts_data = data_list[data_list['method_type'] == "TTS"]
add_check_distances(tts_data)

vc_data = data_list[data_list['method_type'] == "VC"]
add_check_distances(vc_data)

# -------------------------------------------------- #
# -------------------------------------------------- #
# --------------------Checks------------------------ #
# -------------------------------------------------- #
# -------------------------------------------------- #

data_list["check_1"] = data_list["check_distances"].apply(lambda x: x[0])

def check_2(row):
    return row['check_distances'][0] - row['positive_distances'][0]
data_list['check_2'] = data_list.apply(check_2, axis=1)

def check_3(row, number_of_averages):
    try:
        distances = row['check_distances'][:number_of_averages]
        return (sum(distances) / number_of_averages) - max(row['positive_distances'])
    except (IndexError, ValueError):
        return np.nan

def check_4(row, number_of_averages):
    try:
        distances = row['check_distances'][:number_of_averages]
        pos = row['positive_distances'][:number_of_averages]
        return np.mean(distances) - np.mean(pos)
    except (IndexError, ValueError):
        return np.nan

def check_5(row, number_of_averages):
    try:
        distances = row['check_distances'][:number_of_averages]
        pos = row['positive_distances'][:number_of_averages]
        return np.median(distances) - np.median(pos)
    except (IndexError, ValueError):
        return np.nan

def check_6(row, number_of_averages):
    try:
        positive_hits = 0
        for i in range(number_of_averages):
            if row['check_distances'][i] > row['positive_distances'][i]:
                positive_hits += 1
        return positive_hits
    except (IndexError, ValueError):
        return np.nan

for i in range(DISTANCES):
    data_list[f'check_3_{i+1}'] = data_list.apply(lambda row: check_3(row, i+1), axis=1)
    
    data_list[f'check_4_{i+1}'] = data_list.apply(lambda row: check_4(row, i+1), axis=1)
    
    data_list[f'check_5_{i+1}'] = data_list.apply(lambda row: check_5(row, i+1), axis=1)
    
    data_list[f'check_6_{i+1}'] = data_list.apply(lambda row: check_6(row, i+1), axis=1)

# -------------------------------------------------- #
# -------------------------------------------------- #
# --------------------Final Datasets---------------- #
# -------------------------------------------------- #
# -------------------------------------------------- #

data_list_with_vocoder = data_list.copy(deep=True)
data_list_with_vocoder = add_all_checks(data_list_with_vocoder)

tts_data = data_list[(data_list['method_type'] == "bonafide") | (data_list['method_type'] == "TTS")].copy(deep=True)
tts_data = add_all_checks(tts_data)

vc_data = data_list[(data_list['method_type'] == "bonafide") | (data_list['method_type'] == "VC")].copy(deep=True)
vc_data = add_all_checks(vc_data)

data_list = data_list[data_list["method_type"] != "Vocoder"]
data_list = add_all_checks(data_list)

nclass =[]
SAVE_root = SAVE

# -------------------------------------------------- #
# -------------------------------------------------- #
# --------------------Check 1----------------------- #
# -------------------------------------------------- #
# -------------------------------------------------- #
check = "check_1"
SAVE = f"{SAVE_root}/{check}"
create_check_figures(check)

# -------------------------------------------------- #
# -------------------------------------------------- #
# --------------------Check 2----------------------- #
# -------------------------------------------------- #
# -------------------------------------------------- #

check = "check_2"
SAVE = f"{SAVE_root}/{check}"
create_check_figures(check)

# -------------------------------------------------- #
# -------------------------------------------------- #
# --------------------Check 3----------------------- #
# -------------------------------------------------- #
# -------------------------------------------------- #
check = "check_3"
SAVE = f"{SAVE_root}/{check}"
for i in range(DISTANCES):
    checky = f"{check}_{i+1}"
    create_check_figures(checky)
create_viz(check)

# -------------------------------------------------- #
# -------------------------------------------------- #
# --------------------Check 4----------------------- #
# -------------------------------------------------- #
# -------------------------------------------------- #
check = "check_4"
SAVE = f"{SAVE_root}/{check}"
for i in range(DISTANCES):
    checky = f"{check}_{i+1}"
    create_check_figures(checky)
create_viz(check)

# -------------------------------------------------- #
# -------------------------------------------------- #
# --------------------Check 5----------------------- #
# -------------------------------------------------- #
# -------------------------------------------------- #
check = "check_5"
SAVE = f"{SAVE_root}/{check}"
for i in range(DISTANCES):
    checky = f"{check}_{i+1}"
    create_check_figures(checky)
create_viz(check)

# -------------------------------------------------- #
# -------------------------------------------------- #
# --------------------Check 6----------------------- #
# -------------------------------------------------- #
# -------------------------------------------------- #
check = "check_6"
SAVE = f"{SAVE_root}/{check}"
for i in range(DISTANCES):
    checky = f"{check}_{i+1}"
    create_check_figures(checky)
create_viz(check)

SAVE = SAVE_root
# -------------------------------------------------- #
# -------------------------------------------------- #
# --------------------Settings---------------------- #
# -------------------------------------------------- #
# -------------------------------------------------- #

# Sample datasets
settings = {
    "MODEL_PATH": MODEL_PATH,
    "BATCHES": BATCHES,
    "LABELS": LABELS,
    "NUMBER_OF_SPEAKER": NUMBER_OF_SPEAKER,
    "NUMBER_OF_DEEPFAKES": NUMBER_OF_DEEPFAKES,
    "DISTANCES": DISTANCES,
    "MAX_LENGTH": MAX_LENGTH
}

# -------------------------------------------------- #
# -------------------------------------------------- #
# --------------------EERs-------------------------- #
# -------------------------------------------------- #
# -------------------------------------------------- #

eers = [
    ["check_1", False, "Distance between audio to check and a single genuine", data_list["check_1_eer"].iloc[0], data_list["check_1_threshold"].iloc[0]],
    ["check_1", True, "Distance between audio to check and a single genuine", data_list_with_vocoder["check_1_eer"].iloc[0], data_list_with_vocoder["check_1_threshold"].iloc[0]],
    ["check_2", False, "Using a triplet of two genuine audios and the audio to check", data_list["check_2_eer"].iloc[0], data_list["check_2_threshold"].iloc[0]],
    ["check_2", True, "Using a triplet of two genuine audios and the audio to check", data_list_with_vocoder["check_2_eer"].iloc[0], data_list_with_vocoder["check_2_threshold"].iloc[0]],
]

for i in range(DISTANCES):
    eers.append([
        f"check_3_{i+1}",
        False,
        f"Using the maximum distance of {i+1} genuine distances against the mean distance the audio to check against {i+1} genuine audio files",
        data_list[f"check_3_{i+1}_eer"].iloc[0],
        data_list[f"check_3_{i+1}_threshold"].iloc[0]
    ])
    eers.append([
        f"check_3_{i+1}",
        True,
        f"Using the maximum distance of {i+1} genuine distances against the mean distance the audio to check against {i+1} genuine audio files",
        data_list_with_vocoder[f"check_3_{i+1}_eer"].iloc[0],
        data_list_with_vocoder[f"check_3_{i+1}_threshold"].iloc[0]
    ])
for i in range(DISTANCES):
    eers.append([
        f"check_4_{i+1}",
        False,
        f"Using the average distance of {i+1} genuine distances against the mean distance the audio to check against {i+1} genuine audio files",
        data_list[f"check_4_{i+1}_eer"].iloc[0],
        data_list[f"check_4_{i+1}_threshold"].iloc[0]
    ])
    eers.append([
        f"check_4_{i+1}",
        True,
        f"Using the average distance of {i+1} genuine distances against the mean distance the audio to check against {i+1} genuine audio files",
        data_list_with_vocoder[f"check_4_{i+1}_eer"].iloc[0],
        data_list_with_vocoder[f"check_4_{i+1}_threshold"].iloc[0]
    ])
for i in range(DISTANCES):
    eers.append([
        f"check_5_{i+1}",
        False,
        f"Using the median distance of {i+1} genuine distances against the median distance the audio to check against {i+1} genuine audio files",
        data_list[f"check_5_{i+1}_eer"].iloc[0],
        data_list[f"check_5_{i+1}_threshold"].iloc[0]
    ])
    eers.append([
        f"check_5_{i+1}",
        True,
        f"Using the median distance of {i+1} genuine distances against the median distance the audio to check against {i+1} genuine audio files",
        data_list_with_vocoder[f"check_5_{i+1}_eer"].iloc[0],
        data_list_with_vocoder[f"check_5_{i+1}_threshold"].iloc[0]
    ])
for i in range(DISTANCES):
    eers.append([
        f"check_6_{i+1}",
        False,
        f"Using {i+1} positive distances and {i+1} distances from audio to check against positive sample. Counting how many checks are further away than the genuine distances",
        data_list[f"check_6_{i+1}_eer"].iloc[0],
        data_list[f"check_6_{i+1}_threshold"].iloc[0]
    ])
    eers.append([
        f"check_6_{i+1}",
        True,
        f"Using {i+1} positive distances and {i+1} distances from audio to check against positive sample. Counting how many checks are further away than the genuine distances",
        data_list_with_vocoder[f"check_6_{i+1}_eer"].iloc[0],
        data_list_with_vocoder[f"check_6_{i+1}_threshold"].iloc[0]
    ])

# -------------------------------------------------- #
# -------------------------------------------------- #
# --------------------Save Data--------------------- #
# -------------------------------------------------- #
# -------------------------------------------------- #

# Convert to DataFrames
settings = pd.DataFrame(list(settings.items()), columns=["Setting", "Value"])
eers = pd.DataFrame(eers, columns=["Check", "Including Vocoder", "Description", "EER","Threshold"])
nclass = pd.DataFrame(nclass, columns=["Check", "Including Vocoder", "Method name", "Correct","Wrong"])
grouped_nclass = nclass.groupby('Check').apply(lambda x: x).reset_index(drop=True)

# Create a Pandas Excel writer object
with pd.ExcelWriter(f"{SAVE}/analytics.xlsx") as writer:
    eers.to_excel(writer, sheet_name='EER', index=False)    
    grouped_nclass.to_excel(writer, sheet_name='Classification numbers', index=False)
    settings.to_excel(writer, sheet_name='Settings', index=False)


data_list.to_csv(f"{SAVE}/data_list.csv")
data_list.to_excel(f"{SAVE}/data_list.xlsx")

data_list_with_vocoder.to_csv(f"{SAVE}/data_list_with_vocoder.csv")
data_list_with_vocoder.to_excel(f"{SAVE}/data_list_with_vocoder.xlsx")

tts_data.to_csv(f"{SAVE}/tts_data.csv")
tts_data.to_excel(f"{SAVE}/tts_data.xlsx")

vc_data.to_csv(f"{SAVE}/vc_data.csv")
vc_data.to_excel(f"{SAVE}/vc_data.xlsx")
