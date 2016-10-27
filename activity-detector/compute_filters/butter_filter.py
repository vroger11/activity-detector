"""
    Module to allow the user to compute filters on the signal
"""
from scipy.signal import butter, lfilter


def butter_bandpass_filter(signal, low_cut, high_cut, sample_rate, order=5):
    """
        process a bandpass filter on the signal

        :param signal: 1d signal
        :param sample_rate: sample rate of the signal
        :param low_cut: low cut frequency (in Hz)
        :param high_cut: high cut frequency (in Hz)
        :param order:

        :return: filtered signal
    """

    nyq = 0.5 * sample_rate
    low = low_cut / nyq
    high = high_cut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return lfilter(b, a, signal)

def butter_low_pass_filter(signal, sample_rate, low_cut, order=5):
    """
        process a low pass filter on the signal

        :param signal: 1d signal
        :param sample_rate: sample rate of the signal
        :param low_cut: low cut frequency (in Hz)
        :param order:

        :return: filtered signal
    """

    low = low_cut / (0.5 * sample_rate)
    b, a = butter(order, low, btype='lowpass')
    return lfilter(b, a, signal)

def butter_high_pass_filter(signal, sample_rate, high_cut, order=5):
    """
        process a high pass filter on the signal

        :param signal: 1d signal
        :param sample_rate: sample rate of the signal
        :param high_cut: high cut frequency (in Hz)
        :param order:

        :return: filtered signal
    """

    high = high_cut / (0.5 * sample_rate)
    b, a = butter(order, high, btype='highpass')
    return lfilter(b, a, signal)
