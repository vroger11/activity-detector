"""
    Module helping user to plot figures with clusters biside signal, spectrogram
    or only the clusters among time
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

class PlotClusters:
    """
        Class for plotting clusters, it can add sectrograms and signal waveform
    """

    def __init__(self, show_signal=True, show_spectrogram=True, max_frequency=None):
        """
            :param show_signal: show the subplot of the signal
            :param show_spectrogram: show the subplot of the spectrogram
            :param max_frequency: max frequency to display in the spectrogram, if None it will show
                                  the whole spectrogram; Not use with show_audio_with_cluster
        """

        self.show_signal = show_signal
        self.show_spectrogram = show_spectrogram
        self.max_frequency = max_frequency

    def show_clusters(self, signal, sample_rate, clusters):
        """
            plot the signal waveform, spectogram and cluster corresponding to the signal
            cluster must have the same scale as signal

            :param signal: one channel signal
            :param sample_rate: sampling frequency (in Hz)
            :param cluster: matrix of s*n, were s is the clusters an n the of samples of the signal
        """

        figure = self._compute_figure(signal=signal,
                                      sample_rate=sample_rate,
                                      clusters=clusters)
        figure.show()
        plt.show()

    def save_clusters(self, filename_out, signal, sample_rate, clusters):
        """
            compute the figure to show/save

            :param signal: one channel signal
            :param sample_rate: sampling frequency (in Hz)
            :param clusters: matrix of s*n, were s is the clusters an n the of samples of the signal
        """

        figure = self._compute_figure(signal=signal,
                                      sample_rate=sample_rate,
                                      clusters=clusters)

        figure.savefig(filename_out, dpi=200, bbox_inches='tight', papertype='ledger')
        plt.close()


    def _compute_figure(self, signal, sample_rate, clusters):

        total_subplots = 1 + self.show_signal + self.show_spectrogram
        figure, axarr = plt.subplots(total_subplots, sharex=False)
        plt.tight_layout(pad=-0.71)

        id_subplot = 0
        if self.show_signal:
            # plot the waveform
            time = np.linspace(0, len(signal) / sample_rate, num=len(signal))
            axarr[id_subplot].plot(time, signal)
            axarr[id_subplot].set_title('Waveform')
            axarr[id_subplot].set_ylabel("Amplitude")
            axarr[id_subplot].set_xticklabels([])
            id_subplot += 1

        if self.show_spectrogram:
            # plot the spectgram
            [spectrum, freqs, times] = compute_spectrogram(signal, sample_rate)
            if self.max_frequency is None:
                max_frequency = freqs[-1]
                _, index_frequency = spectrum.shape
            else:
                index_frequency = np.argmax(freqs >= self.max_frequency)
                max_frequency = freqs[index_frequency]

            axarr[id_subplot].matshow(spectrum[0:index_frequency+1, :],
                                      origin='lower',
                                      extent=(times[0], times[-1], freqs[0], max_frequency),
                                      aspect='auto')

            axarr[id_subplot].set_title('Spectrogram')
            axarr[id_subplot].set_ylabel("Frequency in Hz")
            axarr[id_subplot].set_xticklabels([])
            axarr[id_subplot].xaxis.grid(which='major', color='Black',
                                         linestyle='-', linewidth=0.25)
            axarr[id_subplot].yaxis.grid(which='major', color='Black',
                                         linestyle='-', linewidth=0.25)
            id_subplot += 1

        # plot cluster
        cluster = np.matrix(clusters)
        dim_x, _ = cluster.shape
        cmap = plt.get_cmap("nipy_spectral")
        axarr[id_subplot].matshow(cluster,
                                  aspect='auto',
                                  origin='lower',
                                  extent=[0, len(signal) / sample_rate, 0, dim_x],
                                  cmap=cmap)
        axarr[id_subplot].set_ylabel("Cluster")
        axarr[id_subplot].set_xlabel("Time in seconds")
        axarr[id_subplot].axes.xaxis.set_ticks_position('bottom')

        # add grid
        axarr[id_subplot].yaxis.set_minor_locator(MultipleLocator(1))

        axarr[id_subplot].xaxis.grid(which='major', color='Black', linestyle='-', linewidth=0.5)
        axarr[id_subplot].yaxis.grid(which='major', color='Black', linestyle='-', linewidth=0.5)
        axarr[id_subplot].yaxis.grid(which='minor', color='Black', linestyle='--', linewidth=0.25)

        return figure


def compute_spectrogram(signal, sample_rate):
    """
        compute spectrogram from signal
        :param signal:
        :return: matrice representing the spectrum, frequencies corresponding and times
    """
    [spectrum, freqs, times] = plt.mlab.specgram(signal, NFFT=1024, Fs=sample_rate,
                                                 noverlap=512, window=np.hamming(1024))
    spectrum = 10. * np.log10(spectrum)

    return [spectrum, freqs, times]

def vector_of_cluster_to_matrix(vector, values_possible=None):
    """
        :param vector: should contains values >= 0
        :param values_possible: a list of possible values in vector

        :return: matrix that can be used in the show_signal_with_cluster function
    """

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
