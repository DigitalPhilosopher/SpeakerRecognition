import os
from dotenv import load_dotenv

load_dotenv()

TRAIN_DATASET = os.path.abspath(os.environ.get("TRAIN_DATASET"))
TEST_DATASET = os.path.abspath(os.environ.get("TEST_DATASET"))

SAMPLERATE = int(os.environ.get("SAMPLERATE"))
N_MFCC = int(os.environ.get("N_MFCC"))
WIN_LENGTH = int(os.environ.get("WIN_LENGTH"))
HOP_LENGTH = int(os.environ.get("HOP_LENGTH"))
WINDOW = os.environ.get("WINDOW")
N_MELS = int(os.environ.get("N_MELS"))
F_MIN = float(os.environ.get("F_MIN"))
f_max_str = os.environ.get("F_MAX")
try:
    F_MAX = float(f_max_str) if f_max_str is not None else None
except ValueError:
    F_MAX = None
FRAME_LENGTH = int(os.environ.get("FRAME_LENGTH"))
FRAME_PADDING = bool(os.environ.get("FRAME_PADDING"))

BATCH_SICE = int(os.environ.get("BATCH_SICE"))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS"))