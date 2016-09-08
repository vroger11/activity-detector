import os
import argparse
import logging.config
import ast
import pickle
import numpy as np
import librosa
from scipy.stats import mstats
from sklearn import metrics
from compute_clusters import gaussian_mixtures
from compute_features.mfcc import FeatureMfcc
import plotting.plot_clusters as plt_clusters
from plotting import plot_internal_indices

def save_obj(obj, filename):
    with open(filename + '.pkl', 'wb') as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)

def load_obj(filename):
    with open(filename + '.pkl', 'rb') as file:
        return pickle.load(file)

def learn_model(folder_audio, feature_extractor, max_learn):
    # get features
    LOGGER.info("Getting features")
    features = []
    file_taken = 0
    for root, dirs, files in os.walk(folder_audio):
        LOGGER.info("Getting features from: " + root)

        for file in files:
            # stop taking files when we reach the limit
            if (not max_learn is None) and file_taken == max_learn:
                break

            path_to_file = os.path.join(root, file)

            try:
                features_file = feature_extractor.get_mfcc_from_file(path_to_file)
            except:
                LOGGER.warning("There is a problem while computing mfcc on: " + path_to_file)
                continue

            features = np.hstack((features, features_file)) if features != [] else features_file
            file_taken += 1

    features = mstats.zscore(features, axis=1, ddof=1)
    # model require n*d; with n the number of observations and d the dimension of an observation
    features = np.transpose(features)

    # learn model
    LOGGER.info("Learning model")
    model = gaussian_mixtures.Model(n_components=30,
                                    n_iter=100,
                                    alpha=.5,
                                    verbose=0,
                                    covariance_type="diag")
    predicted, values_possible = model.learn_model(features)
    LOGGER.info("Done. Converged: " + str(model.dpgmm_model.converged_))

# TODO find better indices performances
    # evaluate silhouette indices
#    silhouette_sample_score = metrics.silhouette_samples(features, predicted, metric='euclidean')
#    silhouette_mean_clusters = np.zeros((1, len(values_possible)))
#    predicted = np.array(predicted)
#    k = 0
#    for i in values_possible:
#        index = np.where(predicted == i)
#        silhouette_mean_clusters[k] = np.mean(silhouette_sample_score[index])
#        k += 1
#
#    return [model, values_possible, silhouette_mean_clusters]
    return [model, values_possible]

def forward_model(folder_out, folder_audio, model, values_possible, feature_extractor):
    signal = []
    sample_rate = 0
    for root, dirs, files in os.walk(folder_audio):
        LOGGER.info("Saving in: " + folder_out)
        for file in files:
            path_to_file = os.path.join(root, file)
            try:
                signal, sample_rate = librosa.load(path_to_file, sr=None)
                features_file = feature_extractor.get_mfcc(signal, sample_rate)
            except Exception as e:
                LOGGER.warning("There is a problem with: " + path_to_file)
                LOGGER.warning(e)
                continue

            features = mstats.zscore(features_file, axis=1, ddof=1)
            features = np.transpose(features)
            clusters = model.predic_clusters(features)
            m_clusters = plt_clusters.vector_of_cluster_to_matrix(
                clusters,
                values_possible=list(values_possible)
                )

            filename_out, _ = os.path.splitext(file)
            path_out = os.path.join(folder_out, filename_out + ".png")

            plt_clusters.save_audio_with_cluster(path_out,
                                                 signal,
                                                 sample_rate,
                                                 m_clusters,
                                                 show_signal=False)

def main(args):
    if not os.path.exists(args.folder_out):
        os.makedirs(args.folder_out)

#    model, values_possible, silhouette_score = learn_model(args.folder_audio,
#                                                           args.freq_min,
#                                                           args.freq_max,
#                                                           args.max_learn)

    feature_extractor = FeatureMfcc(windows=0.06, shift=0.03,
                                    freq_min=args.freq_min,
                                    freq_max=args.freq_max,
                                    n_mfcc=26,
                                    energy=True)

    model, values_possible = learn_model(args.folder_audio,
                                         feature_extractor,
                                         args.max_learn)

    save_obj(model, os.path.join(args.folder_out, 'model'))
    save_obj(values_possible, os.path.join(args.folder_out, 'values_possible'))
#    plot_internal_indices.plot_silhouette(silhouette_score)

    # plot result
    LOGGER.info("Saving results")
    forward_model(args.folder_out,
                  args.folder_audio,
                  model,
                  values_possible,
                  feature_extractor)


if __name__ == '__main__':
    # prepare parser of arguments
    PARSER = argparse.ArgumentParser(description='Process the algorithm to segment the audio')
    PARSER.add_argument('folder_audio', metavar='folder_audio_in', type=str,
                        help='folder containing the audio files')
    PARSER.add_argument('folder_out', metavar='folder_out', type=str,
                        help='folder containing the results')
    PARSER.add_argument('freq_min', metavar='min_frequency', type=int,
                        help='minimum frequency')
    PARSER.add_argument('freq_max', metavar='max_frequency', type=int,
                        help='maximum frequency')
    PARSER.add_argument('-ml', '--max_learn', type=int,
                        help='maximum number of file to take for learning the model.' +
                        ' By default all of them',
                        default=None)
    PARSER.add_argument('-v', '--verbose', action='store_true', help='Show every log')
    PARSER.add_argument('-l', '--logFile', type=str,
                        help='File where the logs will be saved', default=None)

    # parse arguments
    ARGS = PARSER.parse_args()

    # configure logging
    with open('config/logging.json') as config_description:
        config = ast.literal_eval(config_description.read())
        logging.config.dictConfig(config)

    LOGGER = logging.getLogger('activityDetectorDefault')
    if ARGS.verbose:
        LOGGER.setLevel(logging.DEBUG)

    if ARGS.logFile:
        # TODO correct: the logs in the file should be the same as activityDetectorDefault
        LOGGER.addHandler(logging.handlers.RotatingFileHandler(ARGS.logFile))

    main(ARGS)
