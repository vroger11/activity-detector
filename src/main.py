import os
import argparse
import logging.config
import ast
import numpy as np
import librosa
from scipy.stats import mstats
from src.compute_clusters import gaussian_mixtures
from src.compute_features import mfcc
import src.plotting_clusters.plot_informations as plt_clusters

def main():
    # prepare parser of arguments
    parser = argparse.ArgumentParser(description='Process the algorithm to segment the audio')
    parser.add_argument('folder_audio', metavar='folder_audio_in', type=str,
                        help='folder containing the audio files')
    parser.add_argument('folder_out', metavar='folder_out', type=str,
                        help='folder containing the results')
    parser.add_argument('freq_min', metavar='min_frequency', type=int,
                        help='minimum frequency')
    parser.add_argument('freq_max', metavar='max_frequency', type=int,
                        help='maximum frequency')
    parser.add_argument('-ml', '--max_learn', type=int,
                        help='maximum number of file to take for learning the model.' +
                        ' By default all of them',
                        default=None)
    parser.add_argument('-v', '--verbose', action='store_true', help='Show every log')
    parser.add_argument('-l', '--logFile', type=str,
                        help='File where the logs will be saved', default=None)

    # parse arguments
    args = parser.parse_args()

    # configure logging
    with open('config/logging.json') as config_description:
        config = ast.literal_eval(config_description.read())
        logging.config.dictConfig(config)

    logger = logging.getLogger('activityDetectorDefault')
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if args.logFile:
        # TODO correct: the logs in the file should be the same as activityDetectorDefault
        logger.addHandler(logging.handlers.RotatingFileHandler(args.logFile))

    # begin the program
    files = os.listdir(args.folder_audio)

    # get features
    logger.info("Getting features")
    features = []
    file_taken = 0
    for file in files:
        logger.info("Getting features from: " + file)
        path_to_file = os.path.normpath(args.folder_audio + "/" + file)
        try:
            features_file = mfcc.get_mfcc_from_file(path_to_file,
                                                    windows=0.06,
                                                    shift=0.03,
                                                    freq_min=args.freq_min,
                                                    freq_max=args.freq_max,
                                                    n_mfcc=26,
                                                    energy=True)
        except:
            logger.warning("There is a problem while computing mfcc on: " + path_to_file)
            continue

        features = np.hstack((features, features_file)) if features != [] else features_file
        file_taken += 1
        if (not args.max_learn is None) and file_taken == args.max_learn:
            break

    features = mstats.zscore(features, axis=1, ddof=1)
    # model require n*d; with n the number of observations and d the dimension of an observation
    features = np.transpose(features)

    # learn model
    logger.info("Learning model")
    model = gaussian_mixtures.Model(n_components=30,
                                    n_iter=100,
                                    alpha=.5,
                                    verbose=0,
                                    covariance_type="diag")
    model.learn_model(features)
    logger.info("Done. Converged: " + str(model.dpgmm_model.converged_))
    # get number of states
    clusters = model.predic_clusters(features)
    values_possible = np.unique(clusters)

    if not os.path.exists(args.folder_out):
        os.makedirs(args.folder_out)

    # plot result
    logger.info("Saving results")

    signal = []
    sample_rate = 0
    for file in files:
        path_to_file = os.path.join(args.folder_audio, file)
        try:
            signal, sample_rate = librosa.load(path_to_file, sr=None)
            features_file = mfcc.get_mfcc_from_file(path_to_file,
                                                    windows=0.06,
                                                    shift=0.03,
                                                    freq_min=args.freq_min,
                                                    freq_max=args.freq_max,
                                                    n_mfcc=26,
                                                    energy=True)
        except Exception as e:
            logger.warning("There is a problem with: " + path_to_file)
            logger.warning(e)
            continue

        features = mstats.zscore(features_file, axis=1, ddof=1)
        features = np.transpose(features)
        clusters = model.predic_clusters(features)
        m_clusters = plt_clusters.vector_of_cluster_to_matrix(clusters,
                                                              values_possible=list(values_possible))

        filename_out, _ = os.path.splitext(file)
        path_out = os.path.join(args.folder_out, filename_out + ".png")

        logger.info("Saving: " + path_out)
        plt_clusters.save_audio_with_cluster(path_out,
                                             signal,
                                             sample_rate,
                                             m_clusters,
                                             show_signal=False)

if __name__ == '__main__':
    main()
