"""
    module defining a mfcc extractor
"""

import numpy as np
import librosa
from .feature_extractor import FeatureExtractor


class FeatureMfcc(FeatureExtractor):
    """
        Feature extractor of mfcc
    """

    def __init__(self, windows, shift, energy=True, freq_min=1500, freq_max=8000, n_mfcc=13):
        """
        :param windows: > 0 (in seconds)
        :param shift: > 0 (in seconds)
        :param freq_min: lowest frequency (in Hz)
        :param freq_max: highest frequency (in Hz)
        """

        self.windows = windows
        self.shift = shift
        self.energy = energy
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.n_mfcc = n_mfcc

        if energy:
            self.n_mfcc -= 1

    def get_feature(self, signal, sample_rate):
        '''
        compute the mfcc features corresponding to the parameters

        :param signal: one channel signal
        :param sample_rate: sampling frequency (in Hz)
        :return: the mfcc corresponding to all parameters
        '''

        n_fft = round(self.windows * sample_rate)
        hop_length = round(self.shift * sample_rate)

        mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=self.n_mfcc,
                                    fmax=self.freq_max, fmin=self.freq_min, n_fft=n_fft,
                                    hop_length=hop_length, htk=True, n_mels=256)
        if self.energy:
            energy_vec = librosa.feature.rmse(y=signal, n_fft=n_fft, hop_length=hop_length)
            return np.vstack((energy_vec, mfcc))
        else:
            return mfcc
