"""
    module defining an energy extractor
"""

import librosa
from .feature_extractor import FeatureExtractor


class FeatureEnergy(FeatureExtractor):
    """
        Feature extractor of energy
    """

    def __init__(self, windows, shift):
        """
            :param windows: > 0 (in seconds)
            :param shift: > 0 (in seconds)
        """

        self.windows = windows
        self.shift = shift

    def get_feature(self, signal, sample_rate):
        '''
        compute the energy features corresponding to the parameters

        :param signal: one channel signal
        :param sample_rate: sampling frequency (in Hz)
        :return: the eneergy corresponding to all parameters
        '''

        n_fft = round(self.windows * sample_rate)
        hop_length = round(self.shift * sample_rate)

        return librosa.feature.rmse(y=signal, n_fft=n_fft, hop_length=hop_length)
