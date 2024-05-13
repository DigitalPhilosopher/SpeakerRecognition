import torchaudio
import torch
from abc import ABC, abstractmethod

class Frontend(ABC):
    @abstractmethod
    def __init__(self, number_output_parameters=80, sample_rate=16000, max_length=1024):
        """
        Initialize the Frontend with default values.

        :param number_output_parameters: Number of output parameters, default is 80.
        :param sample_rate: The sample rate of the audio waveform, default is 16000 Hz.
        """
        self.number_output_parameters = number_output_parameters
        self.sample_rate = sample_rate
        self.max_length = max_length

    @abstractmethod
    def __call__(self, waveform):
        """
        Process the waveform and produce the output.

        :param waveform: The input audio waveform.
        :return: Processed output.
        """
        pass


class MFCCTransform(Frontend):
    def __init__(self, number_output_parameters=80, sample_rate=16000, max_length=1024):
        super().__init__(number_output_parameters, sample_rate)
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=number_output_parameters,
            log_mels=True
        )

    def __call__(self, waveform):
        # Ensure the output waveform is mono
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        mfcc = self.mfcc_transform(waveform).squeeze(0)

        # Pad or truncate the mfcc to max_length
        if mfcc.size(1) > self.max_length:
            mfcc = mfcc[:, :self.max_length]
        else:
            padding = self.max_length - mfcc.size(1)
            mfcc = torch.nn.functional.pad(mfcc, (0, padding))

        return mfcc