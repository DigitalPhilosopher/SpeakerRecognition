import os
from typing import List, Tuple
import pandas as pd
from .Loader import Loader
from extraction_utils.data_utils import read_label_file


def extract_speaker_id(file_path):
    dir_name = os.path.dirname(file_path)
    dir_parts = dir_name.split(os.path.sep)
    last_folder = dir_parts[-1]
    speaker_id = last_folder.split("_")[-1]
    return speaker_id


class BSILoader(Loader):
    def read_data(self, directory):
        self.data_list: List[Tuple[str, int]] = read_label_file(directory)
        self.data_list = pd.DataFrame(self.data_list, columns=[
                                      "filename", "is_genuine", "method_type", "method_name", "vocoder"])

        self.data_list["utterance"] = self.data_list["filename"].apply(
            lambda x: os.path.basename(x).split(".")[0])
        self.data_list["speaker"] = self.data_list["filename"].apply(
            extract_speaker_id)
