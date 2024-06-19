from .Loader import Loader
import pandas as pd
import os


class VoxCelebLoader(Loader):
    def read_data(self, directory):
        base = f"{os.getcwd()}/../data/VoxCeleb"
        data_list = []
        for set in directory:
            split = set['split']
            dir = f"{base}/{set['name']}/{set['split']}/aac"
            for root, dirs, files in os.walk(dir):
                for file in files:
                    if not file.endswith(".m4a"):
                        continue
                    utterance = os.path.splitext(
                        os.path.basename(os.path.join(root, file)))[0]
                    id = os.path.join(root, file).split("/")[-3]
                    speaker_id = int(id[2:])
                    data_list.append({
                        'filename': os.path.join(root, file),
                        'utterance': utterance,
                        'speaker': speaker_id,
                        'method_type': 'bonafide',
                        'method_name': 'bonafide',
                        'vocoder': 'bonafide',
                        'is_genuine': 1
                    })
        self.data_list = pd.DataFrame(data_list, columns=[
            'filename', 'utterance', 'speaker', 'method_type', 'method_name', 'vocoder', 'is_genuine',])
