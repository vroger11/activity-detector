import numpy as np
import librosa
import matplotlib.pyplot as plt


def show_audio(filename):
    """
    show the waveform and the spectogram on the signal from filename
    :param filename: path to the file containing an audio signal
    :return: None
    """

    # read the audio file
    # plot the waveform
    signal, sample_rate = librosa.load(filename, sr=None)

    _, axarr = plt.subplots(2, sharex=True)
    time = np.linspace(0, len(signal) / sample_rate, num=len(signal))
    axarr[0].plot(time, signal)
    axarr[0].set_title('Waveform')
    axarr[0].set_ylabel("Amplitude")

    # plot the spectgram
    axarr[1].specgram(signal, NFFT=1024, Fs=sample_rate,
                      noverlap=512, window=np.hamming(1024))
    axarr[1].set_title('Spectrogram')
    axarr[1].set_ylabel("Frequency in Hz")
    axarr[1].set_xlabel("Time in seconds")
    plt.show()

def show_audio_with_cluster(signal, sample_rate, cluster, show_signal=True, show_spectrogram=True):
    '''
    plot the signal waveform, spectogram and cluster corresponding to the signal
    cluster must have the same scale as signal

    :param signal: one channel signal
    :param sample_rate: sampling frequency (in Hz)
    :param cluster: matrix of s*n, were s is the clusters an n the of samples of the signal
    :param show_signal: show the subplot of the signal
    :param show_spectrogram: show the subplot of the spectrogram
    '''
    figure = _compute_figure(signal=signal,
                             sample_rate=sample_rate,
                             cluster=cluster,
                             show_signal=show_signal,
                             show_spectrogram=show_spectrogram)
    figure.show()
    plt.show()

def save_audio_with_cluster(filename_out, signal, sample_rate, cluster, show_signal=True, show_spectrogram=True):
    '''
    compute the figure to show/save

    :param signal: one channel signal
    :param sample_rate: sampling frequency (in Hz)
    :param cluster: matrix of s*n, were s is the clusters an n the of samples of the signal
    :param show_signal: show the subplot of the signal
    :param show_spectrogram: show the subplot of the spectrogram
    '''
    figure = _compute_figure(signal=signal,
                             sample_rate=sample_rate,
                             cluster=cluster,
                             show_signal=show_signal,
                             show_spectrogram=show_spectrogram)

    figure.savefig(filename_out, dpi=200, bbox_inches='tight', papertype='ledger')
    plt.close()


def _compute_figure(signal, sample_rate, cluster, show_signal=True, show_spectrogram=True):

    total_subplots = 1 + show_signal + show_spectrogram
    figure, axarr = plt.subplots(total_subplots, sharex=True)

    id_subplot = 0
    if show_signal:
        # plot the waveform
        time = np.linspace(0, len(signal) / sample_rate, num=len(signal))
        axarr[id_subplot].plot(time, signal)
        axarr[id_subplot].set_title('Waveform')
        axarr[id_subplot].set_ylabel("Amplitude")
        id_subplot += 1

    if show_spectrogram:
        # plot the spectgram
        axarr[id_subplot].specgram(signal,
                                   NFFT=1024,
                                   Fs=sample_rate,
                                   noverlap=512,
                                   window=np.hamming(1024))
        axarr[id_subplot].set_title('Spectrogram')
        axarr[id_subplot].set_ylabel("Frequency in Hz")
        id_subplot += 1

    # plot cluster
    m = np.matrix(cluster)
    dim_x, _ = m.shape
    cmap = plt.get_cmap("nipy_spectral")
    axarr[id_subplot].matshow(m,
                              aspect='auto',
                              origin='lower',
                              extent=[0, len(signal) / sample_rate, 0, dim_x],
                              cmap=cmap)
    axarr[id_subplot].set_ylabel("Cluster")
    axarr[id_subplot].set_xlabel("Time in seconds")
    axarr[id_subplot].xaxis.set_ticks_position('bottom')
    # TODO find how to add grid

    return figure

def vector_of_cluster_to_matrix(vector, values_possible=None):
    '''
    :param vector: should contains values >= 0
    :param values_possible: a list of possible values in vector
    :return: matrix that can be used in the show_signal_with_cluster function
    '''

    if values_possible is None:
        values_possible = list(np.unique(vector))
        color_values = 1 / (len(values_possible) + 1)
        clusters = np.ones(shape=(len(values_possible), len(vector)))
    else:
        number_max = len(values_possible)
        color_values = 1 / (number_max + 1)
        clusters = np.ones(shape=(number_max, len(vector)))

    for i in range(len(vector)):
        index = values_possible.index(vector[i])
        clusters[index, i] = color_values * index

    return clusters
