from .Loader import Loader
import pandas as pd
import os


class VoxCelebLoader(Loader):
    def read_data(self, directory):
        base = f"{os.getcwd()}/../data/VoxCeleb"
        data_list = pd.DataFrame()
        for set in directory:
            split = set['split']
            current_df = pd.read_csv(
                f"{base}/{set['name']}/{set['split']}/metadata.csv")
            data_list = pd.concat([data_list, current_df], ignore_index=True)

        self.data_list = pd.DataFrame(data_list, columns=[
            'filename', 'utterance', 'speaker', 'method_type', 'method_name', 'vocoder', 'is_genuine',])
