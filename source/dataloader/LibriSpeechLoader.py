from .Loader import Loader
import pandas as pd
import os


class LibriSpeechLoader(Loader):
    def read_data(self, directory):
        base = "/mnt/e/LibriSpeech/"  # TODO clean this
        data_list = []
        for set in directory:
            split = set['split'].split(".")
            dir = base + split[0] + "-" + set['name']
            if len(split) > 1:
                dir += "-" + split[1]
            for root, dirs, files in os.walk(dir):
                for file in files:
                    if not file.endswith(".flac"):
                        continue
                    utterance = os.path.splitext(
                        os.path.basename(os.path.join(root, file)))[0]
                    speaker_id = int(utterance.split("-")[0])
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
