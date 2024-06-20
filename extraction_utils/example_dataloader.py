from typing import List, Tuple
import os
import numpy as np
import torch
import torch.nn as nn
import torchaudio.functional as F
from torch import Tensor
import librosa
from torch.utils.data import Dataset
from random import randrange
import random
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler

from get_label_files import get_label_files
from data_utils import read_label_file


def pad(x, max_len=64600, random_slice: bool = False):
    x_len = x.shape[0]
    if x_len >= max_len:
        if random_slice:
            start_idx = np.random.randint(0, x_len - max_len + 1)
            return x[start_idx:start_idx + max_len]
        else:
            return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


class Dataset_train(Dataset):
    def __init__(self, label_file_path_list: List[str]):
        self.data_list: List[Tuple[str, int]] = read_label_file(label_file_path_list)
        self.cut = 64600  # take ~4 sec audio (64600 samples)
        method_list = [x[3] for x in self.data_list]
        vocoder_list = [x[4] for x in self.data_list]
        self.method_list = list(set(method_list))
        self.vocoder_list = list(set(vocoder_list))
        self.method_list.sort()
        self.vocoder_list.sort()
        print(f"!!!!!!!Num method:", len(self.method_list))
        print(f"!!!!!!!Num vocoder:", len(self.vocoder_list), self.vocoder_list)

    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, index):
        utt_data = self.data_list[index]
        X, fs = librosa.load(utt_data[0], sr=16000)

        X, index = librosa.effects.trim(X, top_db=35)
        Y = X.astype(np.float32)
        Y = Tensor(Y)
        X_pad = pad(Y, self.cut, random_slice=True)
        x_inp = Tensor(X_pad)

        target = utt_data[1]
        methodtype = utt_data[2]
        method = utt_data[3]
        vocoder = utt_data[4]
        index_of_method = self.method_list.index(method) if method in self.method_list else None
        index_of_vocoder = self.vocoder_list.index(vocoder) if vocoder in self.vocoder_list else None
        mask_method_value = 0 if ("unknown" in method.lower() or "bonafide" in method.lower()) else 1
        mask_vocoder_value = 0 if (
                    "unknown" in vocoder.lower() or "bonafide" in vocoder.lower() or "novocoder" in vocoder.lower()) else 1
        return x_inp, target, index_of_method, mask_method_value, index_of_vocoder, mask_vocoder_value


    def get_method(self, index):
        utt_data = self.data_list[index]
        method = utt_data[3]
        if "Unknown" in method:
            method = utt_data[4]
        return method


class Dataset_eval(Dataset):
    def __init__(self, label_file_path_list: List[str]):
        '''self.list_IDs	: list of strings (each string: utt key),
           '''

        self.data_list: List[Tuple[str, int]] = read_label_file(label_file_path_list)

        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        utt_data = self.data_list[index]
        X, fs = librosa.load(utt_data[0], sr=16000)
        X, index = librosa.effects.trim(X, top_db=35)
        X_pad = pad(X, self.cut, random_slice=True)
        x_inp = Tensor(X_pad)
        return x_inp, utt_data[1]


def get_weighted_train_dataloader(train_set, batch_size:int = 16):
    """
    Returns a train loader which samples spoof and genuine with 50 % probability and
     each spoof method (e.g. VITS, Tacotron, knnvc,...) with the same probability
    Returns:
    """

    method_freq = defaultdict(int)
    for i in range(train_set.__len__()):
        method = train_set.get_method(i)
        method_freq[method] += 1
    total_methods = len(method_freq)
    bonafide_weight = 0.5
    methods_weight = (1 - bonafide_weight) / (total_methods - 1)

    method_weight = {}
    for method, count in method_freq.items():
        if method == "bonafide":
            method_weight[method] = bonafide_weight
        else:
            method_weight[method] = methods_weight
    for method in method_weight:
        method_weight[method] /= method_freq[method]

    weights = []
    for i in range(train_set.__len__()):
        method = train_set.get_method(i)
        weights.append(method_weight[method])

    # Step 4: Use WeightedRandomSampler
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    # Now, you can use the sampler in your DataLoader
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=8, drop_last=True)
    return train_loader


if __name__ == "__main__":

    labels_text_path_list_train, labels_text_path_list_dev, labels_text_path_list_test, all_datasets_used = get_label_files(
                    use_bsi_tts = True,
                    use_bsi_vocoder = True,
                    use_bsi_vc = True,
                    use_bsi_genuine = True,
                    use_bsi_ttsvctk = True,
                    use_bsi_ttslj = True,
                    use_bsi_ttsother = True,
                    use_bsi_vocoderlj = True,
                    use_wavefake = False,
                    use_LibriSeVoc = False,
                    use_lj = False,
                    use_asv2019 = False,
                    )


    train_set = Dataset_train(label_file_path_list=labels_text_path_list_train)
    eval_set = Dataset_eval(label_file_path_list=labels_text_path_list_test)

    train_loader = get_weighted_train_dataloader(train_set, batch_size=16)
    for i, data in enumerate(train_loader):
        x_inp, label, index_of_method, mask_method_value, index_of_vocoder, mask_vocoder_value = data

