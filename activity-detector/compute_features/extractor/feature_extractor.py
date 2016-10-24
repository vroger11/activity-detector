"""
    Module defining what a feature extractor is
"""

import librosa

class FeatureExtractor():
    """
        Class that define what a feature extractor is
    """

    def get_feature_from_file(self, filename):
        """
        :param filename: the path of the audio file

        :return: feature corresponding to all parameters
        """

        try:
            signal, sample_rate = librosa.load(filename, sr=None)
        except:
            raise

        return self.get_feature(signal, sample_rate)

    def get_feature(self, signal, sample_rate):
        """
            Compute the feature from signal corresponding to all parameters.
            Have to be implemented

            :param sample_rate: sampling frequency (in Hz)

            :return: feature corresponding
        """
