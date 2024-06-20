from typing import List, Tuple, Dict
import os
import glob

def create_label_file_fake_or_real(data_path_dict:Dict[str, str]):
    paths_to_check = [
        "f_o_r_testing",
        "f_o_r_training",
        "f_o_r_validation",
    ]
    for subset in paths_to_check:
        if subset not in data_path_dict or (not os.path.exists(data_path_dict[subset])):
            continue

        subset_path = data_path_dict[subset]


        subset_path_real = os.path.join(subset_path, "real")
        subset_path_fake = os.path.join(subset_path, "fake")

        data_list = []
        wav_files_genuine = list(glob.glob(os.path.join(subset_path_real, "*.wav")))
        wav_files_genuine = [x[len(subset_path):] for x in wav_files_genuine]
        wav_files_genuine = [[x, "-", "bonafide"] for x in wav_files_genuine]
        wav_files_spoof = list(glob.glob(os.path.join(subset_path_fake, "*.wav")))
        wav_files_spoof = [x[len(subset_path):] for x in wav_files_spoof]
        wav_files_spoof = [[x, "attack", "spoof"] for x in wav_files_spoof]

        data_list += wav_files_genuine
        data_list += wav_files_spoof


        max_val = 0
        min_val = 0
        if "f_o_r_testing" == subset:
            praefix = "test"
        elif "f_o_r_training" == subset:
            praefix = "train"
        elif "f_o_r_validation" == subset:
            praefix = "valid"

        d_l = data_list
        min_val = max_val

        save_path = os.path.join(subset_path, f"labels-FOR-{praefix}.txt")
        with open(save_path, 'w') as file:
            for tup in d_l:
                line = ','.join(tup)
                file.write(line + "\n")

