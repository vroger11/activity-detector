"""
    Use to forward a model previously learned
"""

import os
import argparse
import logging.config
import ast
import numpy as np
import librosa
import plotting.plot_clusters as plt_clusters
from tools.manipulate_objects import load_obj


def plot_forwarded(folder_out, folder_audio, folder_forwarded, values_possible, freq_max):
    """
        use forwarded clusters and audio to plot the results

        :param folder_out: where the results will be
        :param folder_audio:
        :param folder_forwarded:
        :param values_possible:
        :param freq_max: maximum frequency
    """

    plotter = plt_clusters.PlotClusters(show_signal=False,
                                        show_spectrogram=True,
                                        max_frequency=freq_max)
    if not os.path.exists(folder_out):
        os.makedirs(folder_out)

    LOGGER.info("Saving results")
    for root, _, files in os.walk(folder_audio):
        LOGGER.info("Saving in: " + folder_out)
        for file in files:
            path_to_file = os.path.join(root, file)
            try:
                signal, sample_rate = librosa.load(path_to_file, sr=None,  mono=None)
            except Exception as exception:
                LOGGER.warning("There is a problem with: " + path_to_file)
                LOGGER.warning(exception)
                continue

            # save results
            filename_out, _ = os.path.splitext(file)
            subfolders = root.replace(folder_audio, '')
            path_out_forwarded = os.path.join(folder_forwarded, subfolders)
            path_out_forwarded = os.path.join(path_out_forwarded, filename_out + ".txt")
            clusters = np.loadtxt(path_out_forwarded, delimiter=" ")

            m_clusters = plt_clusters.vector_of_cluster_to_matrix(
                clusters,
                values_possible=list(values_possible)
                )

            path_out_image = os.path.join(folder_out, subfolders)
            if not os.path.exists(path_out_image):
                os.makedirs(path_out_image)

            path_out_image = os.path.join(path_out_image, filename_out + ".png")
            plotter.save_audio_with_cluster(path_out_image,
                                            signal,
                                            sample_rate,
                                            m_clusters)

def main(args):
    # get model
    with open(os.path.join(args.folder_forwarded, 'forward_info.json')) as config_description:
        config_forward = ast.literal_eval(config_description.read())

    values_possible = load_obj(os.path.join(config_forward['model_used'], 'values_possible'))

    # plot result
    plot_forwarded(args.folder_out,
                   config_forward['data_forwarded'],
                   os.path.join(args.folder_forwarded, 'clusters'),
                   values_possible,
                   args.freq_max)

if __name__ == '__main__':
    # prepare parser of arguments
    PARSER = argparse.ArgumentParser(description='Process the algorithm to segment the audio')
    PARSER.add_argument('folder_forwarded', metavar='folder_forwarded', type=str,
                        help='folder containing model + feature extractor')
    PARSER.add_argument('folder_out', metavar='folder_out', type=str,
                        help='folder containing the results')
    PARSER.add_argument('freq_max', metavar='max_frequency', type=int,
                        help='maximum frequency')
    PARSER.add_argument('-v', '--verbose', action='store_true', help='Show every log')
    PARSER.add_argument('-l', '--logFile', type=str,
                        help='File where the logs will be saved', default=None)

    # parse arguments
    ARGS = PARSER.parse_args()

    # configure logging
    with open('config/logging.json') as CONFIG_DESCRIPTION:
        CONFIG_LOG = ast.literal_eval(CONFIG_DESCRIPTION.read())
        logging.config.dictConfig(CONFIG_LOG)

    LOGGER = logging.getLogger('activityDetectorDefault')
    if ARGS.verbose:
        LOGGER.setLevel(logging.DEBUG)

    if ARGS.logFile:
        # TODO correct: the logs in the file should be the same as activityDetectorDefault
        LOGGER.addHandler(logging.handlers.RotatingFileHandler(ARGS.logFile))

    main(ARGS)
