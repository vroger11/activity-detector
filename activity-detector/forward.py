"""
    Use to forward a model previously learned
"""

import os
import argparse
import logging.config
import ast
import numpy as np
import pickle
from scipy.stats import mstats
import librosa
import plotting.plot_clusters as plt_clusters


def load_obj(filename):
    """
        :param filename: where the object is saved
    """
    with open(filename + '.pkl', 'rb') as file:
        return pickle.load(file)

def forward_model(folder_out, folder_audio, model, values_possible, feature_extractor, freq_max):
    if not os.path.exists(os.path.join(folder_out, "figures")):
        os.makedirs(os.path.join(folder_out, "figures"))

    if not os.path.exists(os.path.join(folder_out, "forwarded")):
        os.makedirs(os.path.join(folder_out, "forwarded"))

    LOGGER.info("Saving results")
    sample_rate = 0
    for root, _, files in os.walk(folder_audio):
        LOGGER.info("Saving in: " + folder_out)
        for file in files:
            path_to_file = os.path.join(root, file)
            try:
                signal, sample_rate = librosa.load(path_to_file, sr=None)
                features_file = feature_extractor.get_mfcc(signal, sample_rate)
            except Exception as exception:
                LOGGER.warning("There is a problem with: " + path_to_file)
                LOGGER.warning(exception)
                continue

            features = mstats.zscore(features_file, axis=1, ddof=1)
            features = np.transpose(features)
            clusters = model.predic_clusters(features)

            # save results
            filename_out, _ = os.path.splitext(file)
            path_out_forwarded = os.path.join(folder_out, "forwarded/" + filename_out + ".txt")
            np.savetxt(path_out_forwarded, clusters, delimiter=" ", fmt='%i',)

            m_clusters = plt_clusters.vector_of_cluster_to_matrix(
                clusters,
                values_possible=list(values_possible)
                )

            path_out_image = os.path.join(folder_out, "figures/" + filename_out + ".png")
            plt_clusters.save_audio_with_cluster(path_out_image,
                                                 signal,
                                                 sample_rate,
                                                 m_clusters,
                                                 show_signal=False,
                                                 max_frequency=freq_max)

def main(args):
    # get model
    model = load_obj(os.path.join(args.folder_model, 'model'))
    values_possible = load_obj(os.path.join(args.folder_model, 'values_possible'))
    feature_extractor = load_obj(os.path.join(args.folder_model, 'feature_extractor'))

    # plot result
    forward_model(args.folder_out,
                  args.folder_audio,
                  model,
                  values_possible,
                  feature_extractor,
                  args.freq_max)

if __name__ == '__main__':
    # prepare parser of arguments
    PARSER = argparse.ArgumentParser(description='Process the algorithm to segment the audio')
    PARSER.add_argument('folder_audio', metavar='folder_audio_in', type=str,
                        help='folder containing the audio files')
    PARSER.add_argument('folder_model', metavar='folder_model', type=str,
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
    with open('config/logging.json') as config_description:
        CONFIG_LOG = ast.literal_eval(config_description.read())
        logging.config.dictConfig(CONFIG_LOG)

    LOGGER = logging.getLogger('activityDetectorDefault')
    if ARGS.verbose:
        LOGGER.setLevel(logging.DEBUG)

    if ARGS.logFile:
        # TODO correct: the logs in the file should be the same as activityDetectorDefault
        LOGGER.addHandler(logging.handlers.RotatingFileHandler(ARGS.logFile))

    main(ARGS)
