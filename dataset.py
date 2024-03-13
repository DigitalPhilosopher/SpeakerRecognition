from typing import Tuple
import os
import glob
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from feature_extraction import extract_features, segment_tensor

def get_speaker_utterances(directory_path: os.path) -> pd.DataFrame:
    """
    Generates a pandas DataFrame listing speakers and the paths to their audio files.
    
    This function walks through the directory structure starting at `directory_path`,
    searching for audio files with specified formats (mp3, wav, aac, flac). It assumes
    the first level of subdirectories under `directory_path` represents speaker names.
    Each found audio file's path is listed alongside its corresponding speaker name in the DataFrame.
    
    Parameters:
    - directory_path (str): The path to the directory containing subdirectories for each speaker.
    
    Returns:
    - pd.DataFrame: A DataFrame with two columns:
        - 'speaker': The name of the speaker, derived from the name of the subdirectory.
        - 'file': The full path to the audio file.
    """
    data = []
    audio_formats = ['*.mp3', '*.wav', '*.aac', '*.flac']

    for root, _, _ in os.walk(directory_path):
        for audio_format in audio_formats:
            for filename in glob.glob(os.path.join(root, audio_format)):
                speaker_name = os.path.relpath(root, directory_path).split(os.sep)[0]
                file_path = os.path.join(root, filename)
                data.append([speaker_name, file_path])

    return pd.DataFrame(data, columns=['speaker', 'file'])

class AudioDataset(Dataset):
    """
    A dataset class for audio data that loads utterances and their corresponding speaker labels from a given path,
    extracts features for each audio file, and optionally applies normalization and voice activity detection (VAD).

    Attributes:
        dataset (pd.DataFrame): A DataFrame containing the loaded dataset with columns for file paths and extracted features.

    Parameters:
        path (os.PathLike): The file system path to the dataset directory or file.

    TODOs:
        - Implement caching mechanism to save processed features to disk and load them to avoid recomputation.
        - Consider applying normalization to the features for better model performance.
        - Explore integrating voice activity detection (VAD) to focus on parts of the audio where speech is present.
    """
    def __init__(self, path: os.path):
        self.dataset: pd.DataFrame = get_speaker_utterances(path)

        self.preprocessing()
    
    def preprocessing(self):
        """
        Performs preprocessing on the audio dataset to prepare it for model training or evaluation. This method includes
        several steps: reducing the dataset size for testing, feature extraction, segmenting utterances into smaller
        fixed-length sequences, converting speaker labels to tensors, and optionally applying normalization and
        voice activity detection (VAD).

        The preprocessing steps are as follows:
        1. Reduces the dataset to a smaller subset for testing purposes, selecting only the first two entries per speaker
        for the first five unique speakers found in the dataset.
        2. Applies feature extraction to each audio file in the dataset, storing the extracted features in the 'features'
        column.
        3. Segments each utterance in the dataset into smaller fixed-length sequences to ensure uniform input sizes
        for model processing. The segmented utterances replace the original ones in the dataset.
        4. Converts the 'speaker' column from integers or string identifiers to a PyTorch tensor of long integers, facilitating
        the use of these labels in PyTorch models.
        
        TODOs:
        - Implement caching of processed features to disk to avoid recomputation.
        - Apply normalization to the extracted features to improve model performance.
        - Integrate voice activity detection (VAD) to focus on parts of the audio where speech is present, potentially
        improving the relevance of the extracted features.

        This method updates the dataset in-place, replacing the original 'features' and 'speaker' columns with their
        processed counterparts.

        Parameters:
            None

        Returns:
            None
        """
        # TODO: Can we just save this to disk and only compute if not loaded?
        self.dataset['features'] = self.dataset['file'].apply(extract_features)

        self.segment_utterances()
        self.convert_speaker_labels_to_tensor()

        # TODO: Normalize data?
        # TODO: Voice activity detection (VAD)?

    def segment_utterances(self):
        """
        Segments each utterance in the dataset into smaller fixed-length sequences and updates the dataset
        to contain these segmented utterances. Each original utterance's features are divided into segments, and
        for each segment, a new row is added to the dataset. The new rows retain the speaker and file information
        of the original utterance but replace the features with the segmented features.

        This method is useful for preparing variable-length audio data for models that require fixed-length inputs,
        such as certain types of neural networks. By segmenting the utterances, all inputs to the model can be made
        uniform in length, facilitating batch processing and potentially improving model performance.

        The segmentation process discards any remaining frames that do not fit into a segment of the predefined length,
        ensuring that all segments have exactly the same size. This approach simplifies data handling and model
        training but may result in the loss of some data at the ends of utterances.
        
        Attributes:
            dataset (pd.DataFrame): The dataset being modified, expected to contain columns for 'speaker', 'file',
            and 'features', where 'features' is a tensor of the utterance's audio features.
            
        Returns:
            None: The method updates the dataset in-place, replacing the original dataset with one that contains
            the segmented utterances.
        """
        new_rows = []
        
        for _, row in self.dataset.iterrows():
            segments = segment_tensor(row['features'])
            
            for segment in segments:
                new_row = {'speaker': row['speaker'], 'file': row['file'], 'features': segment}
                new_rows.append(new_row)

        self.dataset = pd.DataFrame(new_rows)


    def convert_speaker_labels_to_tensor(self):
        """
        Converts the 'speaker' column of the dataset from string identifiers to a PyTorch tensor
        of long integers using LabelEncoder for consistent integer encoding. This conversion facilitates
        the use of these labels in PyTorch models, which require tensor inputs.
        
        Updates the 'speaker' column in-place, replacing the original speaker identifiers with their
        corresponding tensor representation.
        """
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(self.dataset['speaker'].tolist())

        self.dataset['speaker'] = torch.tensor(encoded_labels, dtype=torch.long)

    def __len__(self) -> int:
        """Returns the number of items in the dataset."""
        return len(self.dataset)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Retrieves the features and speaker label for the item at the specified index.

        Parameters:
            index (int): The index of the item to retrieve.

        Returns:
            Tuple[torch.Tensor, int]: A tuple containing the extracted features as a tensor and the speaker label as an integer.
        """
        features = self.dataset.iloc[index]['features']
        speaker = self.dataset.iloc[index]['speaker']
        return features, speaker