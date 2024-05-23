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
