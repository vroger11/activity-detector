import numpy as np
import librosa
import matplotlib.pyplot as plt
import sys


def show_signal(filename):
    """
    show the waveform and the spectogram on the signal from filename
    :param filename: path to the file containing an audio signal
    :return: None
    """

    # read the audio file
    signal, fs = librosa.load(filename)

    f, axarr = plt.subplots(2, sharex=True)
    # plot the waveform
    time = np.linspace(0, len(signal) / fs, num=len(signal))
    axarr[0].plot(time, signal)
    axarr[0].set_title('Waveform')
    axarr[0].set_ylabel("Amplitude")

    # plot the spectgram
    spectrum, freqs, t, im = axarr[1].specgram(signal, NFFT=1024, Fs=fs, noverlap=512, window=np.hamming(1024))
    axarr[1].set_title('Spectrogram')
    axarr[1].set_ylabel("Frequency in Hz")
    axarr[1].set_xlabel("Time in seconds")
    plt.show()


def show_signal_with_cluster(signal, fs, cluster):
    '''
    len(signal) must be equal to the second line of cluster.shape
    plot the signal wwaveform, spectogram and cluster corresponding to the signal

    :param signal: one channel signal
    :param fs: sampling frequency
    :param cluster: the cluster corresponding to the signal
    :return: None
    '''

    f, axarr = plt.subplots(3, sharex=True)
    # plot the waveform
    time = np.linspace(0, len(signal) / fs, num=len(signal))
    axarr[0].plot(time, signal)
    axarr[0].set_title('Waveform')
    axarr[0].set_ylabel("Amplitude")

    # plot the spectgram
    spectrum, freqs, t, im = axarr[1].specgram(signal, NFFT=1024, Fs=fs, noverlap=512, window=np.hamming(1024))
    axarr[1].set_title('Spectrogram')
    axarr[1].set_ylabel("Frequency in Hz")

    # plot cluster
    m = np.matrix(cluster)
    axarr[2].matshow(m, aspect='auto', origin='lower', extent=[0, len(signal) / fs, 0, 1])
    axarr[2].set_ylabel("Cluster")
    axarr[2].set_xlabel("Time in seconds")
    axarr[2].xaxis.set_ticks_position('bottom')
    plt.show()


def vector_of_cluster_to_matrix(vec):
    '''
    :param vec: should contains values >= 0
    :return: matrix that can be used in the show_signal_with_cluster function
    '''

    value_max = max(vec)
    clusters = np.zeros(shape=(value_max + 1, len(vec)))
    for i in range(len(vec)):
        clusters[vec[i], i] = 1

    return clusters


if __name__ == '__main__':
    signal, fs = librosa.load(sys.argv[1])
    vec = np.zeros(shape=(len(signal)))

    # to test the function, create fake clusters
    vec[0:len(signal) / 2] = 1
    vec[len(signal) / 2:len(signal) * 3 / 4] = 2
    clusters = vector_of_cluster_to_matrix(vec)

    show_signal_with_cluster(signal, fs, clusters)
