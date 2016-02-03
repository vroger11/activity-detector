import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import sys


def show_signal(filename):
    """
    inputs:
        filename: path to the file containing an audio signal
    effect:
        show the waveform and the spectogram on the signal

    Returns None
    """

    #read the audio file
    (fs, signal) = wav.read(filename)

    f, axarr = plt.subplots(2, sharex=True)
    #plot the waveform
    time=np.linspace(0, len(signal)/fs, num=len(signal))
    axarr[0].plot(time, signal)
    axarr[0].set_title('Waveform')
    axarr[0].set_ylabel("Amplitude")

    #plot the spectgram
    spectrum, freqs, t, im = axarr[1].specgram(signal, NFFT=1024, Fs=fs, noverlap=512, window=np.hamming(1024))
    axarr[1].set_title('Spectrogram')
    axarr[1].set_ylabel("Frequencies in Hz")
    axarr[1].set_xlabel("Time in seconds")
    plt.show()



if __name__ == '__main__':
    show_signal(sys.argv[1])
