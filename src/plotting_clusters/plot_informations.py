import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib.colors as color
import sys


def show_audio(filename):
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


def show_audio_with_cluster(signal, fs, cluster, show_signal=True, show_spectrogram=True):
    '''
    len(signal) must be equal to the second line of cluster.shape
    plot the signal wwaveform, spectogram and cluster corresponding to the signal

    :param signal: one channel signal
    :param fs: sampling frequency
    :param cluster:
    :param show_signal: show the subplot of the signal
    :param show_spectrogram: show the subplot of the spectrogram
    :return: the cluster corresponding to the signal
    '''

    total_sbplots = 1 + show_signal + show_spectrogram
    f, axarr = plt.subplots(total_sbplots, sharex=True)

    id_subplot = 0
    if show_signal:
        # plot the waveform
        time = np.linspace(0, len(signal) / fs, num=len(signal))
        axarr[id_subplot].plot(time, signal)
        axarr[id_subplot].set_title('Waveform')
        axarr[id_subplot].set_ylabel("Amplitude")
        id_subplot += 1

    if show_spectrogram:
        # plot the spectgram
        spectrum, freqs, t, im = axarr[id_subplot].specgram(signal, NFFT=1024, Fs=fs, noverlap=512, window=np.hamming(1024))
        axarr[id_subplot].set_title('Spectrogram')
        axarr[id_subplot].set_ylabel("Frequency in Hz")
        id_subplot += 1

    # plot cluster
    m = np.matrix(cluster)
    dim_x, dim_y = m.shape
    cmap = plt.get_cmap("nipy_spectral")
    axarr[id_subplot].matshow(m, aspect='auto', origin='lower', extent=[0, len(signal) / fs, 0, dim_x], cmap=cmap)
    axarr[id_subplot].set_ylabel("Cluster")
    axarr[id_subplot].set_xlabel("Time in seconds")
    axarr[id_subplot].xaxis.set_ticks_position('bottom')
    #TODO find how to add grid
    plt.show()


def vector_of_cluster_to_matrix(vec, number_max=None):
    '''
    :param vec: should contains values >= 0
    :param number_max:
    :return: matrix that can be used in the show_signal_with_cluster function
    '''

    values_possible = list(np.unique(vec))
    if number_max == None:
        colors_values = 1 / (len(values_possible) + 1)
        clusters = np.ones(shape=(len(values_possible), len(vec)))
    else:
        colors_values = 1/(number_max+1)
        clusters = np.ones(shape=(number_max, len(vec)))

    for i in range(len(vec)):
        index = values_possible.index(vec[i]) if number_max == None else vec[i]
        clusters[index, i] = colors_values*index

    return clusters


if __name__ == '__main__':
    signal, fs = librosa.load(sys.argv[1])
    vec = np.zeros(shape=(len(signal)))

    # to test the function, create fake clusters
    vec[0:len(signal) / 2] = 1
    vec[len(signal) / 2:len(signal) * 3 / 4] = 2
    clusters = vector_of_cluster_to_matrix(vec)

    show_audio_with_cluster(signal, fs, clusters)
