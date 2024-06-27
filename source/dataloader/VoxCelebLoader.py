from .Loader import Loader
import pandas as pd
import os


class VoxCelebLoader(Loader):
    def read_data(self, directory):
        base = f"{os.getcwd()}/../data/VoxCeleb"
        data_list = []
        for set in directory:
            split = set['split']
            if len(data_list):
                data_list += pd.read_csv(
                    f"{base}/{set['name']}/{set['split']}/metadata.csv")
            else:
                data_list = pd.read_csv(
                    f"{base}/{set['name']}/{set['split']}/metadata.csv")

        self.data_list = pd.DataFrame(data_list, columns=[
            'filename', 'utterance', 'speaker', 'method_type', 'method_name', 'vocoder', 'is_genuine',])
