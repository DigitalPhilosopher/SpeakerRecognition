import os
from dotenv import load_dotenv
import ast

load_dotenv()

LOGGING_LEVEL = os.getenv('LOGGING_LEVEL', 'INFO').upper()

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

MODEL = os.environ.get("MODEL")
MODEL_PACKAGE = os.environ.get("MODEL_PACKAGE")
params_str = os.environ.get('MODEL_PARAMS', '{}')
try:
    MODEL_PARAMS = ast.literal_eval(params_str)
    if not isinstance(MODEL_PARAMS, dict):
        raise ValueError("MODEL_PARAMS must be a dictionary.")
except (SyntaxError, ValueError) as e:
    print(f"Error parsing MODEL_PARAMS: {e}")
    MODEL_PARAMS = {}
LOSS = os.environ.get("LOSS")
LOSS_PACKAGE = os.environ.get("LOSS_PACKAGE")
BATCH_SICE = int(os.environ.get("BATCH_SICE"))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS"))