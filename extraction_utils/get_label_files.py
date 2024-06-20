import sys
import os
import glob

sys.path.append(os.path.dirname(__file__))

from basic_utils import  get_config_yml
from data_utils import create_label_files_for_datasets
from label_file_creation.asvspoof2021 import create_label_file_asvspoof2021
from label_file_creation.fakeorreal import create_label_file_fake_or_real


config = get_config_yml()

dataset_base_path = config['dataset_base_path']

bsi_vocoder_dataset = f"{dataset_base_path}/BSI-Dataset/Vocoder_libritts"
bsi_genuine_dataset = f"{dataset_base_path}/BSI-Dataset/Genuine_libritts"
genuine_VCTK = f"{dataset_base_path}/BSI-Dataset/Genuine_VCTK"
bsi_tts_dataset = f"{dataset_base_path}/BSI-Dataset/TTS_libritts"
bsi_ttsvctk_dataset = f"{dataset_base_path}/BSI-Dataset/TTS_VCTK"
bsi_ttslj_dataset = f"{dataset_base_path}/BSI-Dataset/TTS_LJ"
bsi_vc_dataset = f"{dataset_base_path}/BSI-Dataset/VC_libritts"
bsi_othertts_dataset = f"{dataset_base_path}/BSI-Dataset/OtherTTS_Fakes"
bsi_vocoderLJ_dataset = f"{dataset_base_path}/BSI-Dataset/Vocoder_LJ"

wavefake_path = f"{dataset_base_path}/WaveFake"
LibriSeVoc_path = f"{dataset_base_path}/LibriSeVoc"
lj_path = f"{dataset_base_path}/LJSpeech-1.1"

genuine_torsten_dataset = f"{dataset_base_path}/BSI-Dataset/Genuine_Thorsten"
bsi_tts_torsten_dataset = f"{dataset_base_path}/BSI-Dataset/TTS_Thorsten"

asv_spoof_2019_train = f"{dataset_base_path}/2019_ASVspoof/Data/ASVspoof2019_LA_train/flac"
asv_spoof_2019_dev = f"{dataset_base_path}/2019_ASVspoof/Data/ASVspoof2019_LA_dev/flac"
asv_spoof_2019_eval = f"{dataset_base_path}/2019_ASVspoof/Data/ASVspoof2019_LA_eval/flac"

fraunhofer_in_the_wild = f"{dataset_base_path}/Fraunhofer_In_The_Wild"
asv_spoof_2021_la_eval = f"{dataset_base_path}/2021_ASVspoof/ASVspoof2021_LA_eval"

for_testing = f"{dataset_base_path}/for-original/testing"
for_training = f"{dataset_base_path}/for-original/training"
for_validation = f"{dataset_base_path}/for-original/validation"

data_path_dict = {}
data_path_dict["bsi_vocoder_dataset"] = bsi_vocoder_dataset
data_path_dict["bsi_genuine_dataset"] = bsi_genuine_dataset
data_path_dict["genuine_VCTK"] = genuine_VCTK
data_path_dict["bsi_tts_dataset"] = bsi_tts_dataset
data_path_dict["bsi_ttsvctk_dataset"] = bsi_ttsvctk_dataset
data_path_dict["bsi_ttslj_dataset"] = bsi_ttslj_dataset
data_path_dict["bsi_vc_dataset"] = bsi_vc_dataset
data_path_dict["bsi_othertts_dataset"] = bsi_othertts_dataset
data_path_dict["bsi_vocoderLJ_dataset"] = bsi_vocoderLJ_dataset
data_path_dict["wavefake_path"] = wavefake_path
data_path_dict["LibriSeVoc_path"] = LibriSeVoc_path
data_path_dict["lj_path"] = lj_path
data_path_dict["genuine_torsten_dataset"] = genuine_torsten_dataset
data_path_dict["bsi_tts_torsten_dataset"] = bsi_tts_torsten_dataset
data_path_dict["asv_spoof_2019_train"] = asv_spoof_2019_train
data_path_dict["asv_spoof_2019_dev"] = asv_spoof_2019_dev
data_path_dict["asv_spoof_2019_eval"] = asv_spoof_2019_eval
data_path_dict["fraunhofer_in_the_wild"] = fraunhofer_in_the_wild
data_path_dict["asv_spoof_2021_la_eval"] = asv_spoof_2021_la_eval
data_path_dict["f_o_r_testing"] = for_testing
data_path_dict["f_o_r_training"] = for_training
data_path_dict["f_o_r_validation"] = for_validation


def create_label_files():
    create_label_files_for_datasets(data_path_dict)
    create_label_file_asvspoof2021(data_path_dict)
    create_label_file_fake_or_real(data_path_dict)


def get_label_files(use_bsi_tts: bool = True,
                    use_bsi_vocoder: bool = True,
                    use_bsi_vc: bool = True,
                    use_bsi_genuine: bool = True,
                    use_bsi_ttsvctk: bool = True,
                    use_bsi_ttslj: bool = True,
                    use_bsi_ttsother: bool = True,
                    use_bsi_vocoderlj: bool = True,
                    use_wavefake: bool = True,
                    use_LibriSeVoc: bool = True,
                    use_lj: bool = True,
                    use_asv2019: bool = True,
                    ):
    labels_text_path_list_train = []
    labels_text_path_list_dev = []
    labels_text_path_list_test = []

    dataset_list = []
    if use_bsi_vocoder and os.path.exists(bsi_vocoder_dataset):
        dataset_list.append(bsi_vocoder_dataset)
    if use_bsi_genuine and os.path.exists(bsi_genuine_dataset):
        dataset_list.append(bsi_genuine_dataset)
    if use_bsi_genuine and os.path.exists(genuine_torsten_dataset):
        dataset_list.append(genuine_torsten_dataset)
    if use_wavefake and os.path.exists(wavefake_path):
        dataset_list.append(wavefake_path)
    if use_LibriSeVoc and os.path.exists(LibriSeVoc_path):
        dataset_list.append(LibriSeVoc_path)
    if use_lj and os.path.exists(lj_path):
        dataset_list.append(lj_path)
    if use_bsi_vocoderlj and os.path.exists(bsi_vocoderLJ_dataset):
        dataset_list.append(bsi_vocoderLJ_dataset)
    if (use_bsi_ttsvctk or use_bsi_ttsother) and os.path.exists(genuine_VCTK):
        dataset_list.append(genuine_VCTK)

    for labels_text_base_path in dataset_list:
        labels_text_path_list_train += [x for x in list(glob.glob(os.path.join(labels_text_base_path, "*.txt"))) if
                                        "train" in x]
        labels_text_path_list_dev += [x for x in list(glob.glob(os.path.join(labels_text_base_path, "*.txt"))) if
                                      "valid" in x]
        labels_text_path_list_test += [x for x in list(glob.glob(os.path.join(labels_text_base_path, "*.txt"))) if
                                       "test" in x]

    dataset_list_2 = []
    if use_bsi_tts and os.path.exists(bsi_tts_dataset):
        dataset_list_2.append(bsi_tts_dataset)
    if use_bsi_tts and os.path.exists(bsi_tts_torsten_dataset):
        dataset_list_2.append(bsi_tts_torsten_dataset)
    if use_bsi_vc and os.path.exists(bsi_vc_dataset):
        dataset_list_2.append(bsi_vc_dataset)
    if use_bsi_ttsother and os.path.exists(bsi_othertts_dataset):
        dataset_list_2.append(bsi_othertts_dataset)
    if use_bsi_ttsvctk and os.path.exists(bsi_ttsvctk_dataset):
        dataset_list_2.append(bsi_ttsvctk_dataset)
    if use_bsi_ttslj and os.path.exists(bsi_ttslj_dataset):
        dataset_list_2.append(bsi_ttslj_dataset)

    for labels_text_base_path in dataset_list_2:
        labels_text_path_list_train += [x for x in list(glob.glob(os.path.join(labels_text_base_path, "**", "*.txt")))
                                        if "train" in x]
        labels_text_path_list_dev += [x for x in list(glob.glob(os.path.join(labels_text_base_path, "**", "*.txt"))) if
                                      "valid" in x]
        labels_text_path_list_test += [x for x in list(glob.glob(os.path.join(labels_text_base_path, "**", "*.txt"))) if
                                       "test" in x]

    all_datasets_used = dataset_list + dataset_list_2
    if use_asv2019 and os.path.exists(asv_spoof_2019_train):
        all_datasets_used.append(asv_spoof_2019_train)
        labels_text_path_list_train += [x for x in list(glob.glob(os.path.join(asv_spoof_2019_train, "*.txt"))) if
                                        "train" in x]
        labels_text_path_list_dev += [x for x in list(glob.glob(os.path.join(asv_spoof_2019_dev, "*.txt"))) if
                                      "dev" in x]
        labels_text_path_list_test += [x for x in list(glob.glob(os.path.join(asv_spoof_2019_eval, "*.txt"))) if
                                       "eval" in x]

        # labels_text_path_list_train += [x for x in list(glob.glob(os.path.join(asv_spoof_2019_train,  "*.txt"))) if "train" in x]
        # labels_text_path_list_train +=  [x for x in list(glob.glob(os.path.join(asv_spoof_2019_dev,  "*.txt"))) if "dev" in x]
        # labels_text_path_list_train += [x for x in list(glob.glob(os.path.join(asv_spoof_2019_eval,  "*.txt"))) if "eval" in x]

    labels_text_path_list_train = [x for x in labels_text_path_list_train if "label" in os.path.basename(x)]
    labels_text_path_list_dev = [x for x in labels_text_path_list_dev if "label" in os.path.basename(x)]
    labels_text_path_list_test = [x for x in labels_text_path_list_test if "label" in os.path.basename(x)]

    return labels_text_path_list_train, labels_text_path_list_dev, labels_text_path_list_test, all_datasets_used


if __name__ == "__main__":
    create_label_files()