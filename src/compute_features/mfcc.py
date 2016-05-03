import librosa
import sys
import numpy as np
import os


def get_mfcc_from_file(filename, windows, shift, energy=True, freq_min=1500, freq_max=8000, n_mfcc=13):
    '''
    cf get_mfcc for others parameters

    :param filename: the path of the audio file
    :return: the mfcc corresponding to all parameters
    '''

    signal, fs = librosa.load(filename)
    return get_mfcc(signal, fs, windows, shift, energy=energy, freq_min=freq_min, freq_max=freq_max, n_mfcc=n_mfcc)


def get_mfcc(signal, fs, windows, shift, energy=True, freq_min=1500, freq_max=8000, n_mfcc=13):
    '''
    compute the mfcc features corresponding to the parameters

    :param signal: one channel signal
    :param fs: sampling frequency (in Hz)
    :param windows: > 0 (in seconds)
    :param shift: > 0 (in seconds)
    :param freq_min: lowest frequency (in Hz)
    :param freq_max: highest frequency (in Hz)
    :return: the mfcc corresponding to all parameters
    '''

    n_fft = round(windows * fs)
    hop_length = round(shift * fs)

    if energy:
        n_mfcc -= 1
        energy_vec = librosa.feature.rmse(y=signal, n_fft=n_fft, hop_length=hop_length)

    mfcc = librosa.feature.mfcc(y=signal, sr=fs, n_mfcc=n_mfcc, fmax=freq_max, fmin=freq_min, n_fft=n_fft,
                                hop_length=hop_length, htk=True)
    if energy:
        return np.vstack((energy_vec, mfcc))
    else:
        return mfcc


if __name__ == '__main__':
    # TODO use argparse
    if len(sys.argv) != 5 and len(sys.argv) != 7:
        print("usage: " + sys.argv[0] + " <folder audio in> <folder output> <window_features> <hop_time>",
              file=sys.stderr)
        print("or")
        print("usage: " + sys.argv[
            0] + " <folder audio in> <folder output> <window_features> <hop_time> <freq_min> <freq_max>",
              file=sys.stderr)

    if len(sys.argv) == 5:
        freq_min = 1200
        freq_max = 8000
    else:
        freq_min = float(sys.argv[5])
        freq_max = float(sys.argv[6])

    folder = sys.argv[1]
    files = os.listdir(sys.argv[1])

    ##get features
    print("Getting features")
    features = []
    for file in files:
        print("Getting features from: " + file)
        features_file = get_mfcc_from_file(folder + file, windows=float(sys.argv[3]), shift=float(sys.argv[4]),
                                           freq_min=freq_min, freq_max=freq_max)
        # write into a csv file
        np.savetxt(sys.argv[2] + file + ".csv", features_file, delimiter=",")
