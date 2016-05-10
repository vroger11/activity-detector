import librosa
import numpy as np
import os
import argparse


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
    # parse arguments
    parser = argparse.ArgumentParser(description='Process mfcc of some files in a folder')
    parser.add_argument('folder_audio', metavar='folder_audio_in', type=str,
                        help='folder containing the audio files')
    parser.add_argument('folder_output', metavar='folder_output', type=str,
                        help='folder where the resulting files will be put')
    parser.add_argument('windows', metavar='window_features', type=float,
                        help='windows of the mfcc (in seconds)')
    parser.add_argument('shift', metavar='hop_time', type=float,
                        help='hoptime (in seconds)')
    parser.add_argument('freq_min', metavar='frequency_min', type=int,
                        help='minimum frequency (in Hertz) looked at', nargs='?', default=1200)
    parser.add_argument('freq_max', metavar='frequency_max', type=int,
                        help='maximum frequency (in Hertz) looked at', nargs='?', default=8000)

    args = parser.parse_args()
    files = os.listdir(args.folder_audio)

    ##get features
    print("Getting features")
    features = []
    for file in files:
        print("Getting features from: " + file)
        features_file = get_mfcc_from_file(args.folder_audio + file, windows=args.windows, shift=args.shift,
                                           freq_min=args.freq_min, freq_max=args.freq_max)
        # write into a csv file
        np.savetxt(args.folder_output + file + ".csv", features_file, delimiter=",")
