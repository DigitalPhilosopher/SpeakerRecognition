import argparse
import os
import subprocess

parser = argparse.ArgumentParser(description='Model Setup Parameters')
parser.add_argument('--dir', type=str, required=True, help='Model directory', default="models/")
parser.add_argument('--batches', type=int, default=8, help='Number of batches')
parser.add_argument('--labels', type=str, default='valid', help='Labels (dev/train/test/valid)')
parser.add_argument('--num_speakers', type=int, default=0, help='Number of speakers')
parser.add_argument('--num_deepfakes', type=int, default=10, help='Number of deepfakes')
parser.add_argument('--distances', type=int, default=5, help='Distance value')
parser.add_argument('--max_length', type=int, default=32000, help='Maximum length')
parser.add_argument('--dataset', type=str, default='BSI.deepfake', help='Dataset name (BSI.deepfake/ASVspoof5/inthewild)')

args = parser.parse_args()

DIR = args.dir
BATCHES = args.batches
LABELS = args.labels
NUMBER_OF_SPEAKER = args.num_speakers
NUMBER_OF_DEEPFAKES = args.num_deepfakes
DISTANCES = args.distances
MAX_LENGTH = args.max_length
DATASET = args.dataset

# List to store all .pth files
pth_files = []

# Walk through all directories and files
for root, dirs, files in os.walk(DIR):
    for file in files:
        if file.endswith('.pth'):
            pth_files.append(os.path.join(root, file))

for model_path in pth_files:
    print(f'\n+++++++++++++++++++++\n\nStarting with model {model_path}')
    directory, model = os.path.split(model_path)
    directory = directory + "/"

    if DATASET == "BSI.deepfake":
        execution_file = "source/Analytics Sophisticated.py"
    elif DATASET == "ASVspoof5":
        execution_file = "source/Analytics Sophisticated ASVspoof.py"
    elif DATASET == "inthewild":
        execution_file = "source/Analytics Sophisticated in the wild.py"
    else:
        print("Dataset must be either inthewild, ASVspoof5 or BSI.deepfake")
        exit(-1)
    subprocess.run(["python", execution_file,
        "--model", model,
        "--dir", directory,
        "--batches", str(BATCHES),
        "--labels", LABELS,
        "--num_speakers", str(NUMBER_OF_SPEAKER),
        "--num_deepfakes", str(NUMBER_OF_DEEPFAKES),
        "--distances", str(DISTANCES),
        "--max_length", str(MAX_LENGTH),
        "--dataset", DATASET
    ])
