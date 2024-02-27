import os
import numpy as np
import soundfile
import librosa
import torch
from torch.nn.functional import pad
from config import SAMPLERATE, N_MFCC, WIN_LENGTH, HOP_LENGTH, WINDOW, N_MELS, F_MIN, F_MAX, FRAME_LENGTH, FRAME_PADDING

def extract_features(audio_file: os.path) -> np.ndarray:
    """
    Extracts MFCC features from an audio file.

    This function reads an audio file using soundfile.read, optionally converts it to mono using librosa.to_mono
    if the audio is not already a single channel, and resamples it to a target sample rate if necessary using
    librosa.core.resample. It then calculates and returns the Mel-frequency cepstral coefficients (MFCC) of the audio
    signal using librosa.feature.mfcc.

    Parameters:
    - audio_file (str): The path to the audio file from which features are to be extracted.

    Returns:
    - np.ndarray: A numpy array containing the MFCC features of the audio file.
    """
    wav, samplerate = soundfile.read(audio_file)
    if wav.shape != 2:
        wav = librosa.to_mono(wav.transpose())
    
    if samplerate != SAMPLERATE:
        print(samplerate)
        print(SAMPLERATE)
        wav = librosa.core.resample(y=wav, orig_sr=samplerate, target_sr=SAMPLERATE)

    mfcc = librosa.feature.mfcc(
        y = wav,
        sr = SAMPLERATE,
        n_mfcc = N_MFCC,
        win_length = WIN_LENGTH,
        hop_length = HOP_LENGTH,
        window = WINDOW,
        n_mels = N_MELS,
        fmin = F_MIN,
        fmax = F_MAX
    )

    return torch.tensor(mfcc, dtype=torch.float)


def segment_tensor(input_tensor: torch.Tensor) -> list:
    """
    Segments a given tensor into multiple slices of a specified length, optionally padding the last segment if it
    does not reach the specified length and frame_padding is True.

    Parameters:
    - input_tensor (torch.Tensor): The input tensor to be segmented, expected shape [num_features, sequence_length].
    - segment_length (int): The length of each segment.
    - frame_padding (bool): If True, the last segment will be padded to match segment_length if it is shorter.
    - padding_value (float): The value used for padding the last segment if frame_padding is True.

    Returns:
    - list: A list of segmented tensors, each with shape [num_features, segment_length]. The last segment may be
      padded if frame_padding is True and it is shorter than segment_length.
    """
    num_segments = input_tensor.shape[1] // FRAME_LENGTH
    remainder = input_tensor.shape[1] % FRAME_LENGTH
    
    segments = []
    
    for i in range(num_segments):
        start = i * FRAME_LENGTH
        end = start + FRAME_LENGTH
        segment = input_tensor[:, start:end]
        segments.append(segment)
    
    if FRAME_PADDING and remainder > 0:
        last_segment = input_tensor[:, -remainder:]
        padding_needed = FRAME_LENGTH - last_segment.shape[1]
        last_segment_padded = pad(last_segment, (0, padding_needed))
        segments.append(last_segment_padded)
    
    return segments