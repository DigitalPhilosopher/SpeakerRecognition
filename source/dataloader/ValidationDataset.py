from dataloader import AudioDataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np

class ValidationDatasetFromSet(AudioDataset):
    def __init__(self, loader, valid_set, max_length: int = 0):
        super().__init__(loader, max_length)
        
        self.valid_list = []
        for index, row in valid_set.iterrows():
            anchor_data = self.data_list[
                (self.data_list["utterance"] == row["utterance"]) & 
                (self.data_list["method_name"] == row["method_name"])
            ].iloc[0]

            # Find the corresponding entry in data_list for the "other"
            try:
                other_data = self.data_list[
                    (self.data_list["utterance"] == row["utterance_to_check"]) & 
                    (self.data_list["method_name"] == row["method_name_to_check"])
                ].iloc[0]
            except:
                other_data = self.data_list[
                    (self.data_list["utterance"] == row["utterance_to_check"])
                ].iloc[0]

            # Construct the anchor info dictionary
            anchor_info = {
                "utterance": row["utterance"],
                "method_name": row["method_name"],
                "filename": anchor_data["filename"],
                "speaker": anchor_data["speaker"],
                "is_genuine": True
            }

            # Construct the other info dictionary
            other_info = {
                "utterance": row["utterance_to_check"],
                "method_name": row["method_name_to_check"],
                "filename": other_data["filename"],
                "speaker": other_data["speaker"],
                "is_genuine": False
            }

            # Append the constructed dictionaries to the data_list
            self.valid_list.extend([anchor_info, other_info])

    def __getitem__(self, idx):
        anchor_data, other_data = self.valid_list.iloc[idx]
        return (
            self.read_audio(anchor_data["filename"]), 
            anchor_data["speaker"], 
            anchor_data["utterance"], 
            anchor_data["is_genuine"], 
            anchor_data["method_name"]
        ), (
            self.read_audio(other_data["filename"]), 
            other_data["speaker"], 
            other_data["utterance"], 
            other_data["is_genuine"], 
            other_data["method_name"])

    def __len__(self):
        return len(self.valid_list)

class ValidationDataset(AudioDataset):
    def __getitem__(self, idx):
        anchor_data = self.data_list.iloc[idx]
        return (self.read_audio(anchor_data["filename"]), anchor_data["speaker"], anchor_data["utterance"], anchor_data["is_genuine"], anchor_data["method_name"])

    def __len__(self):
        return len(self.data_list)


def collate_valid_fn(batch):
    audio_data, speakers, utterances, is_genuine, method_name = zip(*batch)
    audio_data = pad_sequence(audio_data, batch_first=True, padding_value=0)
    return audio_data, np.array(speakers), np.array(utterances), np.array(is_genuine), np.array(method_name)

def collate_double_valid_fn(batch):
    # Unzip the batch into separate components for anchor and other
    anchors, others = zip(*batch)

    # Separate each component (audio, speaker, etc.) for anchors
    anchor_audio_data, anchor_speakers, anchor_utterances, anchor_is_genuine, anchor_method_names = zip(*anchors)
    # Separate each component (audio, speaker, etc.) for others
    other_audio_data, other_speakers, other_utterances, other_is_genuine, other_method_names = zip(*others)

    # Pad the audio data sequences for anchors and others
    anchor_audio_data = pad_sequence(anchor_audio_data, batch_first=True, padding_value=0)
    other_audio_data = pad_sequence(other_audio_data, batch_first=True, padding_value=0)

    # Return the collated batches as arrays (or tensors if necessary)
    return (
        anchor_audio_data, 
        np.array(anchor_speakers), 
        np.array(anchor_utterances), 
        np.array(anchor_is_genuine), 
        np.array(anchor_method_names)
    ), (
        other_audio_data, 
        np.array(other_speakers), 
        np.array(other_utterances), 
        np.array(other_is_genuine), 
        np.array(other_method_names)
    )

