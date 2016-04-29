import librosa
import sys
import numpy


def get_mfcc(signal, fs, windows, shift, freq_min=1000, freq_max=8000, n_mfcc=13):
    '''
    compute the mfcc features corresponding to the parameters

    :param signal:
    :param fs: sampling frequency in Hz
    :param windows: >0 in seconds
    :param shift: >0 in seconds
    :param freq_min:
    :param freq_max:
    :return: the mfcc corresponding
    '''
    n_fft = round(windows * fs)
    hop_length = round(shift * fs)
    mfcc = librosa.feature.mfcc(y=signal, sr=fs, n_mfcc=n_mfcc, fmax=freq_max, fmin=freq_min, n_fft=n_fft,
                                hop_length=hop_length, htk=True)
    return mfcc


if __name__ == '__main__':
    if len(sys.argv) != 5 and len(sys.argv) != 7:
        print("usage: " + sys.argv[0] + " <audio filename> <filename output> <window_features> <hop_time>",
              file=sys.stderr)
        print("or")
        print("usage: " + sys.argv[
            0] + " <audio filename> <filename output> <window_features> <hop_time> <freq_min> <freq_max>",
              file=sys.stderr)

    if len(sys.argv) == 5:
        freq_min = 1200
        freq_max = 8000
    else:
        freq_min = float(sys.argv[5])
        freq_max = float(sys.argv[6])

    signal, fs = librosa.load(sys.argv[1])
    mfcc = get_mfcc(signal, fs, float(sys.argv[3]), float(sys.argv[4]), freq_min, freq_max)
    # write into a csv file
    numpy.savetxt(sys.argv[2], mfcc, delimiter=",")
