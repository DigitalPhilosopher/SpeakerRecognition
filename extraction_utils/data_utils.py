import random
from typing import List, Tuple, Dict

import glob
import os



def read_label_file(file_path_list: List[str]) -> List[Tuple[str, int, str, str, str]]:
    """
    Read label files.

    This function reads a list of label files. Each file contains lines with
    audio file paths (relative) and their corresponding labels. The function returns a list
    of tuples containing the full path to each audio file and its integer label.

    Parameters:
    - file_path_list (List[str]): List of paths to label files to be read.

    Returns:
    - List[Tuple[str, int]]: List of tuples with audio file path and its label (0 for "spoof", 1 otherwise).

    """
    data_list = []

    for file_path in file_path_list:
        if "ASV" in file_path:
            data_list += read_label_file_asvspoof([file_path])
            continue
        with open(file_path, 'r') as file:
            dir_path = os.path.dirname(file_path)
            bn = os.path.basename(file_path)
            if "===" in bn:
                method_type = bn.split("===")[1]
                method_name = bn.split("===")[2]
                vocoder_name = bn.split("===")[3].split(".txt")[0]
                if vocoder_name == method_name:
                    method_name = "Unknown"
                if vocoder_name.lower() == "novocoder":
                    vocoder_name = method_name # lets count unknown vocoder as unique vocoder
            else:
                print("!!!!!!Unknown:", bn)
                method_type = "Unknown"
                method_name = "Unknown"
                vocoder_name = "Unknown"
            for line in file:
                try:
                    file_name, _, label = line.strip().split(",")
                except:
                    print("!!!!!!!!!!ERROR with line:", line)
                    print("File: ", file_path)
                if "/" == file_name[0]:
                    file_name = file_name[1:]
                label_int = 0 if label == "spoof" else 1
                # if label_int == 1 and label != "bonafide":
                #     print(f"!!!!!!!! label: {label} ==> {label_int}")
                if label_int == 1:
                    method_type = "bonafide"
                    method_name = "bonafide"
                    vocoder_name = "bonafide"
                data_list.append((os.path.join(dir_path, file_name), label_int, method_type, method_name, vocoder_name))
    # print("!!!!!file_path_list:", file_path_list)
    print(f"number of 0: {sum([1 for x in data_list if x[1] == 0])}")
    print(f"number of 1: {sum([1 for x in data_list if x[1] == 1])}")

    return data_list

def read_label_file_asvspoof(file_path_list: List[str]) -> List[Tuple[str, int, str, str, str]]:
    """
    Read label files.

    This function reads a list of label files. Each file contains lines with
    audio file paths (relative) and their corresponding labels. The function returns a list
    of tuples containing the full path to each audio file and its integer label.

    Parameters:
    - file_path_list (List[str]): List of paths to label files to be read.

    Returns:
    - List[Tuple[str, int]]: List of tuples with audio file path and its label (0 for "spoof", 1 otherwise).

    """
    data_list = []

    for file_path in file_path_list:
        with open(file_path, 'r') as file:
            dir_path = os.path.dirname(file_path)
            bn = os.path.basename(file_path)


            for line in file:
                file_name, mn, label = line.strip().split(",")
                if "/" == file_name[0]:
                    file_name = file_name[1:]
                label_int = 0 if label == "spoof" else 1

                method_type = mn
                method_name = mn
                vocoder_name = "Unknown"

                if label_int == 1:
                    method_type = "bonafide"
                    method_name = "bonafide"
                    vocoder_name = "bonafide"

                data_list.append((os.path.join(dir_path, file_name), label_int, method_type, method_name, vocoder_name))
    # print("!!!!!file_path_list:", file_path_list)
    print(f"number of 0: {sum([1 for x in data_list if x[1] == 0])}")
    print(f"number of 1: {sum([1 for x in data_list if x[1] == 1])}")

    return data_list


def create_training_txt_vocoder(genine_labels_path: str,
                                audio_base_path: str,
                                save_base_path: str = None,
                                method_type:str = None,
                                name_of_method:str = None,
                                vocoder_name:str = None) -> None:
    """
    Create a training text file for vocoders.

    This function reads genuine labels and generates a training text file for
    vocoders based on the audio files present in the given directory.

    Parameters:
    - genine_labels_path (str): Path to the genuine labels file.
    - audio_base_path (str): Base directory containing audio files.

    Returns:
    - None
    """
    genuine_list: List[Tuple[str, int]] = read_label_file([genine_labels_path])


    data_list = []
    method_name = os.path.basename(audio_base_path)
    method_prefix = os.path.basename(os.path.dirname(audio_base_path))
    for genuine_audio_path, _, _, _, _ in genuine_list:
        speaker_name = os.path.basename(os.path.dirname(genuine_audio_path))
        bn = os.path.basename(genuine_audio_path)
        audio_path = os.path.join(audio_base_path, speaker_name, bn)
        # print(audio_path)
        if os.path.exists(audio_path):
            relative_path = os.path.join(method_name, speaker_name, bn)
            if save_base_path is not None:

                diff = audio_base_path[len(save_base_path):]
                if diff[0] == "/":
                    diff = diff[1:]
                relative_path = os.path.join(diff, speaker_name, bn)
            data_list.append([relative_path, method_prefix + "_" + method_name, "spoof"])
        else:
            continue

    if "train" in genine_labels_path:
        praefix = "train"
    elif "test" in genine_labels_path:
        praefix = "test"
    elif "valid" in genine_labels_path:
        praefix = "valid"
    if save_base_path is None:
        save_base_path = os.path.dirname(audio_base_path)

    if method_type is None:
        method_type = praefix
    if name_of_method is None:
        name_of_method = method_name
    if vocoder_name is None:
        vocoder_name = method_name

    save_path = os.path.join(save_base_path, f"labels-BSI-{praefix}==={method_type}==={name_of_method}==={vocoder_name}.txt")
    with open(save_path, 'w') as file:
        for tup in data_list:
            line = ','.join(tup)
            file.write(line + '\n')




def create_training_txt_vc(genine_labels_path: str,
                                audio_base_path: str,
                                save_base_path: str = None,
                                method_type:str = None,
                                name_of_method:str = None,
                                vocoder_name:str = None) -> None:
    """
    Create a training text file for voice conversion dataset.

    This function reads genuine labels and generates a training text file for
    vc based on the audio files present in the given directory.

    Parameters:
    - genine_labels_path (str): Path to the genuine labels file.
    - audio_base_path (str): Base directory containing audio files.

    Returns:
    - None
    """
    genuine_list: List[Tuple[str, int]] = read_label_file([genine_labels_path])


    data_list = []
    method_name = os.path.basename(audio_base_path)
    method_prefix = os.path.basename(os.path.dirname(audio_base_path))
    source_target_names = [os.path.basename(x) for x in list(glob.glob(os.path.join(audio_base_path, "*")))]
    speaker_name_list = [os.path.basename(os.path.dirname(genuine_audio_path)) for genuine_audio_path, _, _, _, _ in genuine_list]
    speaker_name_list = list(set(speaker_name_list))
    for speaker_name in speaker_name_list:
        audio_dir_list = [x for x in source_target_names if x.split("__")[1] == speaker_name]
        for speaker_names in audio_dir_list:
            for audio_path in glob.glob(os.path.join(audio_base_path, speaker_names, "*.wav")):
                bn = os.path.basename(audio_path)
                if os.path.exists(audio_path):
                    relative_path = os.path.join(method_name, speaker_names, bn)
                    if save_base_path is not None:
                        diff = audio_base_path[len(save_base_path):]
                        if diff[0] == "/":
                            diff = diff[1:]
                        relative_path = os.path.join(diff, speaker_names, bn)
                    data_list.append([relative_path, method_prefix + "_" + method_name, "spoof"])
                else:
                    continue

    # for genuine_audio_path, _, _, _, _ in genuine_list:
    #     speaker_name = os.path.basename(os.path.dirname(genuine_audio_path))
    #     bn = os.path.basename(genuine_audio_path)
    #     audio_path = os.path.join(audio_base_path, speaker_name, bn)
    #     # print(audio_path)
    #     if os.path.exists(audio_path):
    #         relative_path = os.path.join(method_name, speaker_name, bn)
    #         if save_base_path is not None:
    #
    #             diff = audio_base_path[len(save_base_path):]
    #             if diff[0] == "/":
    #                 diff = diff[1:]
    #             relative_path = os.path.join(diff, speaker_name, bn)
    #         data_list.append([relative_path, method_prefix + "_" + method_name, "spoof"])
    #     else:
    #         continue

    if "train" in genine_labels_path:
        praefix = "train"
    elif "test" in genine_labels_path:
        praefix = "test"
    elif "valid" in genine_labels_path:
        praefix = "valid"
    if save_base_path is None:
        save_base_path = os.path.dirname(audio_base_path)

    if method_type is None:
        method_type = praefix
    if name_of_method is None:
        name_of_method = method_name
    if vocoder_name is None:
        vocoder_name = method_name

    save_path = os.path.join(save_base_path, f"labels-BSI-{praefix}==={method_type}==={name_of_method}==={vocoder_name}.txt")
    # print("!!!!!!save_path:,", save_path)
    with open(save_path, 'w') as file:
        for tup in data_list:
            line = ','.join(tup)
            file.write(line + '\n')







def create_training_txt_VCTK(audio_base_path: str,
                                save_base_path: str = None,
                                method_type:str = None,
                                name_of_method:str = None,
                                tts_name:str = None, dataset_split:Tuple = (0.9, 0.05, 0.05)) -> None:

    wav_path: List[str] = list(glob.glob(os.path.join(audio_base_path, "*", "*.wav")))
    wav_path.sort()

    prefix_list = ["train", "test", "valid"]
    data_list = {
        "train": [],
        "test": [],
        "valid": [],
    }
    method_name = os.path.basename(audio_base_path)
    method_prefix = os.path.basename(os.path.dirname(audio_base_path))
    speaker_list = list(set([os.path.basename(os.path.dirname(audio_path)) for audio_path in wav_path]))
    speaker_list.sort()
    train_max_idx = int(len(speaker_list) * dataset_split[0])
    test_max_idx = int(len(speaker_list) * (dataset_split[0] + dataset_split[1]))
    if len(speaker_list) > 1:
        train_speaker_list = speaker_list[:train_max_idx]
        test_speaker_list = speaker_list[train_max_idx:test_max_idx]
        valid_speaker_list = speaker_list[test_max_idx:]
    else:
        train_speaker_list= speaker_list
        test_speaker_list = []
        valid_speaker_list = []
    for audio_path in wav_path:
        speaker_name = os.path.basename(os.path.dirname(audio_path))
        if speaker_name in train_speaker_list:
            praefix = "train"
        elif speaker_name in test_speaker_list:
            praefix = "test"
        else:
            praefix = "valid"

        bn = os.path.basename(audio_path)
        audio_path = os.path.join(audio_base_path, speaker_name, bn)
        # print(audio_path)
        if os.path.exists(audio_path):
            relative_path = os.path.join(method_name, speaker_name, bn)
            if save_base_path is not None:

                diff = audio_base_path[len(save_base_path):]
                if diff[0] == "/":
                    diff = diff[1:]
                relative_path = os.path.join(diff, speaker_name, bn)
            data_list[praefix].append([relative_path, method_prefix + "_" + method_name, "spoof"])
        else:
            continue

    for praefix in data_list:
        if len(data_list[praefix]) == 0:
            continue

        if save_base_path is None:
            save_base_path = os.path.dirname(audio_base_path)

        if method_type is None:
            method_type = praefix
        if name_of_method is None:
            name_of_method = method_name
        if tts_name is None:
            tts_name = method_name

        save_path = os.path.join(save_base_path, f"labels-BSI-{praefix}==={method_type}==={name_of_method}==={tts_name}.txt")
        # print("!!!!!!save_path:,", save_path)
        with open(save_path, 'w') as file:
            for tup in data_list[praefix]:
                line = ','.join(tup)
                file.write(line + '\n')



def create_training_txt_wavefake(wavefake_dir_path: str,
                                train_portion:float = 0.9,
                                dev_portion:float = 0.05,
                                test_portion:float = 0.05,
                                label:str = "spoof",
                                dataset_name:str = "Wavefake"
                                 ) -> None:
    """
    Create a training text file for wavefake dataset.
    Parameters:
    - wavefake_dir_path (str): Base directory containing wavefake data

    Returns:
    - None
    """

    methods_to_use = [
        "jsut__multiband_melgan",
        "jsut__parallel_wavegan",
        "ljspeech__full_band_melgan",
        "ljspeech__hifiGAN",
        "ljspeech__melgan",
        "ljspeech__melgan_large",
        "ljspeech__multi_band_melgan",
        "ljspeech__parallel_wavegan",
        "ljspeech__waveglow"
    ]
    method_type = "Vocoder"

    for method_dir in glob.glob(os.path.join(wavefake_dir_path, "*")):
        if not os.path.isdir(method_dir) or os.path.basename(method_dir) not in methods_to_use:
            continue
        print("method_dir", method_dir)
        vocoder_name = os.path.basename(method_dir).split("__")[-1]
        method_name = vocoder_name

        data_list = []
        method_prefix = method_name
        for audio_path in glob.glob(os.path.join(method_dir, "*.wav")):
            relative_path = os.path.join(os.path.basename(method_dir), os.path.basename(audio_path))
            data_list.append([relative_path, method_name, label])

        max_val = 0
        min_val = 0
        for praefix in ["train", "test", "valid"]:
            if praefix == "train":
                max_val += int(len(data_list)*train_portion)
            elif praefix == "test":
                max_val += int(len(data_list)*test_portion)
            else:
                max_val += int(len(data_list)*dev_portion)
            d_l = data_list[min_val: max_val]
            min_val = max_val


            save_path = os.path.join(wavefake_dir_path, f"labels-{dataset_name}-{praefix}==={method_type}==={method_name}==={vocoder_name}.txt")
            # print("!!!!!!save_path:,", save_path)
            with open(save_path, 'w') as file:
                for tup in d_l:
                    line = ','.join(tup)
                    file.write(line + '\n')



def create_training_txt_VocoderLJ(lj_base_path: str,
                                train_portion:float = 0.9,
                                dev_portion:float = 0.05,
                                test_portion:float = 0.05,
                                label:str = "spoof",
                                dataset_name:str = "bsi"
                                 ) -> None:
    """
    Create a training text file for LibriSeVoc dataset.
    Parameters:
    - lj_base_path (str): Base directory containing LibriSeVoc data

    Returns:
    - None
    """
    method_type = "Vocoder"

    for method_dir in glob.glob(os.path.join(lj_base_path, "*")):
        if not os.path.isdir(method_dir):
            continue
        print("method_dir", method_dir)
        vocoder_name = os.path.basename(method_dir)
        method_name = vocoder_name

        data_list = []
        method_prefix = method_name
        for audio_path in glob.glob(os.path.join(method_dir, "*", "*.wav")):
            speaker_name = os.path.basename(os.path.dirname(audio_path))
            # relative_path = os.path.join(method_name, os.path.basename(audio_path))
            relative_path = os.path.join(method_name, speaker_name, os.path.basename(audio_path))
            data_list.append([relative_path, method_name, label])

        max_val = 0
        min_val = 0
        for praefix in ["train", "test", "valid"]:
            if praefix == "train":
                max_val += int(len(data_list)*train_portion)
            elif praefix == "test":
                max_val += int(len(data_list)*test_portion)
            else:
                max_val += int(len(data_list)*dev_portion)
            d_l = data_list[min_val: max_val]
            min_val = max_val


            save_path = os.path.join(lj_base_path, f"labels-{dataset_name}-{praefix}==={method_type}==={method_name}==={vocoder_name}.txt")
            # print("!!!!!!save_path:,", save_path)
            with open(save_path, 'w') as file:
                for tup in d_l:
                    line = ','.join(tup)
                    file.write(line + '\n')


def create_training_txt_LibriSeVoc(wavefake_dir_path: str,
                                train_portion:float = 0.9,
                                dev_portion:float = 0.05,
                                test_portion:float = 0.05,
                                label:str = "spoof",
                                dataset_name:str = "LibriSeVoc"
                                 ) -> None:
    """
    Create a training text file for LibriSeVoc dataset.
    Parameters:
    - wavefake_dir_path (str): Base directory containing LibriSeVoc data

    Returns:
    - None
    """
    methods_to_use = [
        "diffwave",
        "melgan",
        "parallel_wave_gan",
        "wavegrad",
        "wavenet",
        "wavernn"
    ]

    method_type = "Vocoder"

    for method_dir in glob.glob(os.path.join(wavefake_dir_path, "*")):
        if not os.path.isdir(method_dir) or os.path.basename(method_dir) not in methods_to_use:
            continue
        print("method_dir", method_dir)
        vocoder_name = os.path.basename(method_dir)
        method_name = vocoder_name

        data_list = []
        method_prefix = method_name
        for audio_path in glob.glob(os.path.join(method_dir, "*.wav")):
            relative_path = os.path.join(method_name, os.path.basename(audio_path))
            data_list.append([relative_path, method_name, label])

        max_val = 0
        min_val = 0
        for praefix in ["train", "test", "valid"]:
            if praefix == "train":
                max_val += int(len(data_list)*train_portion)
            elif praefix == "test":
                max_val += int(len(data_list)*test_portion)
            else:
                max_val += int(len(data_list)*dev_portion)
            d_l = data_list[min_val: max_val]
            min_val = max_val


            save_path = os.path.join(wavefake_dir_path, f"labels-{dataset_name}-{praefix}==={method_type}==={method_name}==={vocoder_name}.txt")
            # print("!!!!!!save_path:,", save_path)
            with open(save_path, 'w') as file:
                for tup in d_l:
                    line = ','.join(tup)
                    file.write(line + '\n')

def create_training_txt_fraunhofer_in_the_wild(dataset_dir_path: str,
                                dataset_name:str = "fraunhofer_in_the_wild"
                                 ) -> None:
    """
    Create a training text file for LibriSeVoc dataset.
    Parameters:
    - wavefake_dir_path (str): Base directory containing LibriSeVoc data

    Returns:
    - None
    """
    metadata_path = os.path.join(dataset_dir_path, "l_original.txt")
    with open(metadata_path, 'r') as file:
        metadata_lines = file.readlines()

    data_list = []

    for line in metadata_lines:
        # 0.wav,attack,spoof
        # 0.wav,Alec Guinness,spoof
        if len(line.split(",")) != 3:
            continue
        wav_name, speaker_name, spoof_label = line.split(",")
        relative_path = os.path.join("release_in_the_wild", wav_name)
        method_name = "attack" if "spoof" in spoof_label  else "-"
        spoof_label = "spoof" if "spoof" in spoof_label  else "bonafide"
        data_list.append([relative_path, method_name, spoof_label])

    max_val = 0
    min_val = 0
    praefix = "test"

    d_l = data_list
    min_val = max_val

    save_path = os.path.join(dataset_dir_path, f"labels-{dataset_name}-{praefix}.txt")
    with open(save_path, 'w') as file:
        for tup in d_l:
            line = ','.join(tup)
            file.write(line + "\n")


def create_training_txt_LJSpeech(lj_dir_path: str,
                                train_portion:float = 0.1,
                                dev_portion:float = 0.05,
                                test_portion:float = 0.05,
                                label:str = "bonafide"
                                 ) -> None:
    """
    Create a training text file for LJ dataset.
    Parameters:
    - lj_dir_path (str): Base directory containing LJ data

    Returns:
    - None
    """

    data_list = []
    for audio_path in glob.glob(os.path.join(lj_dir_path, "wavs", "*.wav")):
        relative_path = os.path.join("wavs", os.path.basename(audio_path))
        data_list.append([relative_path, "_", label])

    max_val = 0
    min_val = 0
    for praefix in ["train", "test", "valid"]:
        if praefix == "train":
            max_val += int(len(data_list)*train_portion)
        elif praefix == "test":
            max_val += int(len(data_list)*test_portion)
        else:
            max_val += int(len(data_list)*dev_portion)
        d_l = data_list[min_val: max_val]
        min_val = max_val


        save_path = os.path.join(lj_dir_path, f"labels-LJ-{praefix}_genuine.txt")
        with open(save_path, 'w') as file:
            for tup in d_l:
                line = ','.join(tup)
                file.write(line + '\n')




def create_training_txt_vctk(vctk_dir_path: str,
                                train_portion:float = 0.9,
                                dev_portion:float = 0.05,
                                test_portion:float = 0.05,
                                label:str = "bonafide"
                                 ) -> None:
    """
    Create a training text file for vctk dataset.
    Parameters:
    - vctk_dir_path (str): Base directory containing vctk data

    Returns:
    - None
    """

    data_list = []
    for audio_path in glob.glob(os.path.join(vctk_dir_path, "*", "*.wav")):
        speaker_name = os.path.basename(os.path.dirname(audio_path))
        relative_path = os.path.join(speaker_name, os.path.basename(audio_path))
        data_list.append([relative_path, "_", label])

    max_val = 0
    min_val = 0
    for praefix in ["train", "test", "valid"]:
        if praefix == "train":
            max_val += int(len(data_list)*train_portion)
        elif praefix == "test":
            max_val += int(len(data_list)*test_portion)
        else:
            max_val += int(len(data_list)*dev_portion)
        d_l = data_list[min_val: max_val]
        min_val = max_val


        save_path = os.path.join(vctk_dir_path, f"labels-vctk-{praefix}_genuine.txt")
        with open(save_path, 'w') as file:
            for tup in d_l:
                line = ','.join(tup)
                file.write(line + '\n')




def shuffle_text_file(input_file, output_file):
    random.seed(42)

    # Read lines from the input file
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Shuffle the lines
    random.shuffle(lines)

    # Write the shuffled lines to the output file
    with open(output_file, 'w') as f:
        f.writelines(lines[:int(0.9*len(lines))])
    with open(output_file.replace(".txt", "") + "_short.txt", 'w') as f:
        f.writelines(lines[int(0.9*len(lines)):])

def create_label_files_for_datasets(dataset_path_dict:Dict[str, str]):
    training_types = ["train", "valid", "test"]
    genuine_dataset_path = dataset_path_dict["bsi_genuine_dataset"] if "bsi_genuine_dataset" in dataset_path_dict else None
    bsi_vocoder_dataset = dataset_path_dict["bsi_vocoder_dataset"] if "bsi_vocoder_dataset" in dataset_path_dict else None
    genuine_VCTK = dataset_path_dict["genuine_VCTK"] if "genuine_VCTK" in dataset_path_dict else None
    bsi_tts_dataset = dataset_path_dict["bsi_tts_dataset"] if "bsi_tts_dataset" in dataset_path_dict else None
    bsi_tts_torsten_dataset = dataset_path_dict["bsi_tts_torsten_dataset"] if "bsi_tts_torsten_dataset" in dataset_path_dict else None
    bsi_ttsvctk_dataset = dataset_path_dict["bsi_ttsvctk_dataset"] if "bsi_ttsvctk_dataset" in dataset_path_dict else None
    bsi_ttslj_dataset = dataset_path_dict["bsi_ttslj_dataset"] if "bsi_ttslj_dataset" in dataset_path_dict else None
    bsi_vc_dataset = dataset_path_dict["bsi_vc_dataset"] if "bsi_vc_dataset" in dataset_path_dict else None
    bsi_othertts_dataset = dataset_path_dict["bsi_othertts_dataset"] if "bsi_othertts_dataset" in dataset_path_dict else None
    bsi_vocoderLJ_dataset = dataset_path_dict["bsi_vocoderLJ_dataset"] if "bsi_vocoderLJ_dataset" in dataset_path_dict else None
    wavefake_path = dataset_path_dict["wavefake_path"] if "wavefake_path" in dataset_path_dict else None
    LibriSeVoc_path = dataset_path_dict["LibriSeVoc_path"] if "LibriSeVoc_path" in dataset_path_dict else None
    lj_path = dataset_path_dict["lj_path"] if "lj_path" in dataset_path_dict else None
    asv_spoof_2019_train = dataset_path_dict["asv_spoof_2019_train"] if "asv_spoof_2019_train" in dataset_path_dict else None
    asv_spoof_2019_dev = dataset_path_dict["asv_spoof_2019_dev"] if "asv_spoof_2019_dev" in dataset_path_dict else None
    asv_spoof_2019_eval = dataset_path_dict["asv_spoof_2019_eval"] if "asv_spoof_2019_eval" in dataset_path_dict else None
    fraunhofer_in_the_wild = dataset_path_dict["fraunhofer_in_the_wild"] if "fraunhofer_in_the_wild" in dataset_path_dict else None

    if bsi_vocoder_dataset is not None and os.path.exists(bsi_vocoder_dataset):
        labels_text_base_path_base = bsi_vocoder_dataset
        label_text_base_path_list = list(glob.glob(os.path.join(labels_text_base_path_base, "*")))
        label_text_base_path_list = [x for x in label_text_base_path_list if os.path.isdir(x)]
        for labels_text_base_path_spoof in label_text_base_path_list:
            for train_type in training_types:
                labels_text_base_path = genuine_dataset_path
                genuine_path = [x for x in list(glob.glob(os.path.join(labels_text_base_path, "*.txt"))) if train_type in x][0]
                create_training_txt_vocoder(genuine_path, labels_text_base_path_spoof,
                                                method_type="Vocoder",
                                                name_of_method=None,
                                                vocoder_name=None)
    if bsi_tts_dataset is not None and os.path.exists(bsi_tts_dataset):
        tts_path = bsi_tts_dataset
        labels_text_base_path_base_list = list(glob.glob(os.path.join(tts_path, "*")))
        for labels_text_base_path_base in labels_text_base_path_base_list:
            label_text_base_path_list = list(glob.glob(os.path.join(labels_text_base_path_base, "wavs", "*")))
            label_text_base_path_list = [x for x in label_text_base_path_list if os.path.isdir(x)]
            print(label_text_base_path_list)
            for labels_text_base_path_spoof in label_text_base_path_list:
                tts_method_name = os.path.basename(labels_text_base_path_base)
                vocoder_name = os.path.basename(labels_text_base_path_spoof)

                for train_type in training_types:
                    labels_text_base_path = genuine_dataset_path
                    genuine_path = [x for x in list(glob.glob(os.path.join(labels_text_base_path, "*.txt"))) if train_type in x][0]
                    create_training_txt_vocoder(genuine_path, labels_text_base_path_spoof,
                                                save_base_path=labels_text_base_path_base,
                                                method_type="TTS",
                                                name_of_method=tts_method_name,
                                                vocoder_name=vocoder_name)


    if bsi_ttsvctk_dataset is not None and os.path.exists(bsi_ttsvctk_dataset):
        tts_path = bsi_ttsvctk_dataset
        labels_text_base_path_base_list = list(glob.glob(os.path.join(tts_path, "*")))
        for labels_text_base_path_base in labels_text_base_path_base_list:
            label_text_base_path_list = list(glob.glob(os.path.join(labels_text_base_path_base, "wavs", "*")))
            label_text_base_path_list = [x for x in label_text_base_path_list if os.path.isdir(x)]
            print(label_text_base_path_list)
            for labels_text_base_path_spoof in label_text_base_path_list:
                tts_method_name = os.path.basename(labels_text_base_path_base)
                tts_name = os.path.basename(labels_text_base_path_spoof)

                for train_type in training_types:
                    create_training_txt_VCTK(labels_text_base_path_spoof,
                                                save_base_path=labels_text_base_path_base,
                                                method_type="TTS",
                                                name_of_method=tts_method_name,
                                                tts_name=tts_name)


    if bsi_ttslj_dataset is not None and os.path.exists(bsi_ttslj_dataset):
        tts_path = bsi_ttslj_dataset
        labels_text_base_path_base_list = list(glob.glob(os.path.join(tts_path, "*")))
        for labels_text_base_path_base in labels_text_base_path_base_list:
            label_text_base_path_list = list(glob.glob(os.path.join(labels_text_base_path_base, "wavs", "*")))
            label_text_base_path_list = [x for x in label_text_base_path_list if os.path.isdir(x)]
            print(label_text_base_path_list)
            for labels_text_base_path_spoof in label_text_base_path_list:
                tts_method_name = os.path.basename(labels_text_base_path_base)
                tts_name = os.path.basename(labels_text_base_path_spoof)

                for train_type in training_types:
                    create_training_txt_VCTK(labels_text_base_path_spoof,
                                                save_base_path=labels_text_base_path_base,
                                                method_type="TTS",
                                                name_of_method=tts_method_name,
                                                tts_name=tts_name)
    if bsi_othertts_dataset is not None and os.path.exists(bsi_othertts_dataset):
        tts_path = bsi_othertts_dataset
        labels_text_base_path_base_list = list(glob.glob(os.path.join(tts_path, "*")))
        for labels_text_base_path_base in labels_text_base_path_base_list:
            label_text_base_path_list = list(glob.glob(os.path.join(labels_text_base_path_base, "wavs", "*")))
            label_text_base_path_list = [x for x in label_text_base_path_list if os.path.isdir(x)]
            print(label_text_base_path_list)
            for labels_text_base_path_spoof in label_text_base_path_list:
                tts_method_name = os.path.basename(labels_text_base_path_base)
                tts_name = os.path.basename(labels_text_base_path_spoof)

                for train_type in training_types:
                    create_training_txt_VCTK(labels_text_base_path_spoof,
                                                save_base_path=labels_text_base_path_base,
                                                method_type="TTS",
                                                name_of_method=tts_method_name,
                                                tts_name=tts_name)

    if bsi_tts_torsten_dataset is not None and os.path.exists(bsi_tts_torsten_dataset):
        tts_path = bsi_tts_torsten_dataset
        labels_text_base_path_base_list = list(glob.glob(os.path.join(tts_path, "*")))
        for labels_text_base_path_base in labels_text_base_path_base_list:
            label_text_base_path_list = list(glob.glob(os.path.join(labels_text_base_path_base, "wavs", "*")))
            label_text_base_path_list = [x for x in label_text_base_path_list if os.path.isdir(x)]
            print(label_text_base_path_list)
            for labels_text_base_path_spoof in label_text_base_path_list:
                tts_method_name = os.path.basename(labels_text_base_path_base)
                tts_name = os.path.basename(labels_text_base_path_spoof)

                for train_type in training_types:
                    create_training_txt_VCTK(labels_text_base_path_spoof,
                                                save_base_path=labels_text_base_path_base,
                                                method_type="TTS",
                                                name_of_method=tts_method_name,
                                                tts_name=tts_name)

    if bsi_vc_dataset is not None and os.path.exists(bsi_vc_dataset):
        tts_path = bsi_vc_dataset
        labels_text_base_path_base_list = list(glob.glob(os.path.join(tts_path, "*")))
        for labels_text_base_path_base in labels_text_base_path_base_list:
            label_text_base_path_list = list(glob.glob(os.path.join(labels_text_base_path_base, "wavs", "*")))
            label_text_base_path_list = [x for x in label_text_base_path_list if os.path.isdir(x)]
            print(label_text_base_path_list)
            for labels_text_base_path_spoof in label_text_base_path_list:
                tts_method_name = os.path.basename(labels_text_base_path_base)
                vocoder_name = os.path.basename(labels_text_base_path_spoof)

                for train_type in training_types:
                    labels_text_base_path = genuine_dataset_path
                    genuine_path = [x for x in list(glob.glob(os.path.join(labels_text_base_path, "*.txt"))) if train_type in x][0]
                    create_training_txt_vc(genuine_path, labels_text_base_path_spoof,
                                                save_base_path=labels_text_base_path_base,
                                                method_type="VC",
                                                name_of_method=tts_method_name,
                                                vocoder_name=vocoder_name)


    if wavefake_path is not None and os.path.exists(wavefake_path):
        create_training_txt_wavefake(wavefake_path)

    if bsi_vocoderLJ_dataset is not None and os.path.exists(bsi_vocoderLJ_dataset):
        create_training_txt_VocoderLJ(bsi_vocoderLJ_dataset)

    if lj_path is not None and os.path.exists(lj_path):
        create_training_txt_LJSpeech(lj_path)

    if genuine_VCTK is not None and os.path.exists(genuine_VCTK):
        create_training_txt_vctk(genuine_VCTK)

    if LibriSeVoc_path is not None and os.path.exists(LibriSeVoc_path):
        create_training_txt_LibriSeVoc(LibriSeVoc_path)

    if fraunhofer_in_the_wild is not None and os.path.exists(fraunhofer_in_the_wild):
        create_training_txt_fraunhofer_in_the_wild(fraunhofer_in_the_wild)



if __name__ == "__main__":
    training_types = ["train", "valid", "test"]
    dataset_base_path = "/datadisk/ThesisDatasets"


    genuine_dataset_path = f"{dataset_base_path}/BSI-Dataset/Genuine"

    # for train_type in training_types:
    #     labels_text_base_path = f"{dataset_base_path}/BSI-Dataset/Genuine"
    #     genuine_path = [x for x in list(glob.glob(os.path.join(labels_text_base_path, "*.txt"))) if train_type in x][0]
    #     shuffle_text_file(genuine_path, genuine_path.replace(".txt", "")+"_random.txt")
    # import sys
    # sys.exit()
    #


    labels_text_base_path_base = f"{dataset_base_path}/BSI-Dataset/Vocoder"
    label_text_base_path_list = list(glob.glob(os.path.join(labels_text_base_path_base, "*")))
    label_text_base_path_list = [x for x in label_text_base_path_list if os.path.isdir(x)]

    for labels_text_base_path_spoof in label_text_base_path_list:
        print("??????????????labels_text_base_path_spoof:", labels_text_base_path_spoof)
        for train_type in training_types:
            labels_text_base_path = genuine_dataset_path
            genuine_path = [x for x in list(glob.glob(os.path.join(labels_text_base_path, "*.txt"))) if train_type in x][0]
            create_training_txt_vocoder(genuine_path, labels_text_base_path_spoof,
                                            method_type="Vocoder",
                                            name_of_method=None,
                                            vocoder_name=None)



    tts_path = f"{dataset_base_path}/BSI-Dataset/TTS_part/"
    labels_text_base_path_base_list = list(glob.glob(os.path.join(tts_path, "*")))
    for labels_text_base_path_base in labels_text_base_path_base_list:
        label_text_base_path_list = list(glob.glob(os.path.join(labels_text_base_path_base, "wavs", "*")))
        label_text_base_path_list = [x for x in label_text_base_path_list if os.path.isdir(x)]
        print(label_text_base_path_list)
        for labels_text_base_path_spoof in label_text_base_path_list:
            tts_method_name = os.path.basename(labels_text_base_path_base)
            vocoder_name = os.path.basename(labels_text_base_path_spoof)

            for train_type in training_types:
                labels_text_base_path = genuine_dataset_path
                genuine_path = [x for x in list(glob.glob(os.path.join(labels_text_base_path, "*.txt"))) if train_type in x][0]
                create_training_txt_vocoder(genuine_path, labels_text_base_path_spoof,
                                            save_base_path=labels_text_base_path_base,
                                            method_type="TTS",
                                            name_of_method=tts_method_name,
                                            vocoder_name=vocoder_name)
    print(list(glob.glob(os.path.join(tts_path, "**", "*.txt"))))

    tts_path = f"{dataset_base_path}/BSI-Dataset/TTS_VCTK/"
    labels_text_base_path_base_list = list(glob.glob(os.path.join(tts_path, "*")))
    for labels_text_base_path_base in labels_text_base_path_base_list:
        label_text_base_path_list = list(glob.glob(os.path.join(labels_text_base_path_base, "wavs", "*")))
        label_text_base_path_list = [x for x in label_text_base_path_list if os.path.isdir(x)]
        print(label_text_base_path_list)
        for labels_text_base_path_spoof in label_text_base_path_list:
            tts_method_name = os.path.basename(labels_text_base_path_base)
            tts_name = os.path.basename(labels_text_base_path_spoof)

            for train_type in training_types:
                create_training_txt_VCTK(labels_text_base_path_spoof,
                                            save_base_path=labels_text_base_path_base,
                                            method_type="TTS",
                                            name_of_method=tts_method_name,
                                            tts_name=tts_name)

    tts_path = f"{dataset_base_path}/BSI-Dataset/TTS_LJ/"
    labels_text_base_path_base_list = list(glob.glob(os.path.join(tts_path, "*")))
    for labels_text_base_path_base in labels_text_base_path_base_list:
        label_text_base_path_list = list(glob.glob(os.path.join(labels_text_base_path_base, "wavs", "*")))
        label_text_base_path_list = [x for x in label_text_base_path_list if os.path.isdir(x)]
        print(label_text_base_path_list)
        for labels_text_base_path_spoof in label_text_base_path_list:
            tts_method_name = os.path.basename(labels_text_base_path_base)
            tts_name = os.path.basename(labels_text_base_path_spoof)

            for train_type in training_types:
                create_training_txt_VCTK(labels_text_base_path_spoof,
                                            save_base_path=labels_text_base_path_base,
                                            method_type="TTS",
                                            name_of_method=tts_method_name,
                                            tts_name=tts_name)
    print(list(glob.glob(os.path.join(tts_path, "**", "*.txt"))))


    tts_path = f"{dataset_base_path}/BSI-Dataset/VC/"
    labels_text_base_path_base_list = list(glob.glob(os.path.join(tts_path, "*")))
    for labels_text_base_path_base in labels_text_base_path_base_list:
        label_text_base_path_list = list(glob.glob(os.path.join(labels_text_base_path_base, "wavs", "*")))
        label_text_base_path_list = [x for x in label_text_base_path_list if os.path.isdir(x)]
        print(label_text_base_path_list)
        for labels_text_base_path_spoof in label_text_base_path_list:
            tts_method_name = os.path.basename(labels_text_base_path_base)
            vocoder_name = os.path.basename(labels_text_base_path_spoof)

            for train_type in training_types:
                labels_text_base_path = genuine_dataset_path
                genuine_path = [x for x in list(glob.glob(os.path.join(labels_text_base_path, "*.txt"))) if train_type in x][0]
                create_training_txt_vc(genuine_path, labels_text_base_path_spoof,
                                            save_base_path=labels_text_base_path_base,
                                            method_type="VC",
                                            name_of_method=tts_method_name,
                                            vocoder_name=vocoder_name)
    print(list(glob.glob(os.path.join(tts_path, "**", "*.txt"))))

    wavefake_path = f"{dataset_base_path}/WaveFake/Data_to_use"
    create_training_txt_wavefake(wavefake_path)

    labels_text_base_path_base = f"{dataset_base_path}/BSI-Dataset/Vocoder_LJ"
    create_training_txt_VocoderLJ(labels_text_base_path_base)

    lj_path = f"{dataset_base_path}/WaveFake/LJSpeech-1.1"
    create_training_txt_LJSpeech(lj_path)

    wavefake_path = f"{dataset_base_path}/LibriSeVoc/Data_to_use"
    create_training_txt_LibriSeVoc(wavefake_path)