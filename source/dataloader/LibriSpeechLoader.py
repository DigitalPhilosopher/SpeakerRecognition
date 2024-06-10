from .Loader import Loader
import pandas as pd
from datasets import load_dataset, concatenate_datasets


class LibriSpeechLoader(Loader):
    def read_data(self, directory):
        librispeech = []
        if not directory == []:
            datasets = []
            for data in directory:
                dataset = load_dataset(
                    'DigitalPhilosopher/librispeech_asr', data["name"], split=data["split"], trust_remote_code=True)
                datasets.append(dataset)
            librispeech = concatenate_datasets(datasets)
        self.data_list = [{
            'filename': entry['audio']['path'],
            'utterance': entry['id'],
            'speaker': entry['speaker_id'],
            'method_type': 'bona_fide',
            'method_name': 'bona_fide',
            'vocoder': 'bona_fide',
            'is_genuine': 1
        } for entry in librispeech]
        self.data_list = pd.DataFrame(self.data_list, columns=[
                                      'filename', 'utterance', 'speaker', 'method_type', 'method_name', 'vocoder', 'is_genuine',])
