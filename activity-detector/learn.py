"""
    Use to learn a model
"""

import os
import argparse
import logging.config
import ast
import pickle
import random
import numpy as np
from scipy.stats import mstats
#from sklearn import metrics
from compute_clusters import gaussian_mixtures
from compute_features.mfcc import FeatureMfcc
#from plotting import plot_internal_indices

def save_obj(obj, filename):
    """
        :param obj: object to save in a pickle file
        :param filename: where the object will be saved
    """
    with open(filename + '.pkl', 'wb') as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)

def learn_model(folder_audio, feature_extractor, max_learn):
    """
        :param folder_audio: folder where the audios are
        :param feature_extractor: object use to extract features
        :param max_learn: between 0 and 1, help to define the number of files
                          use in every folder. 1 means every file is used

        :return: model learned
    """
    # get features
    LOGGER.info("Getting features")
    features = []
    file_taken = []
    for root, _, files in os.walk(folder_audio):
        LOGGER.info("Getting features from: " + root)

        if (not max_learn is None):
            number_of_file = int(len(files)*max_learn)
            random.shuffle(files)
            files_to_learn = files[0:number_of_file]
        else:
            files_to_learn = files

        file_taken.append(files_to_learn)
        for file in files_to_learn:
            path_to_file = os.path.join(root, file)

            try:
                features_file = feature_extractor.get_mfcc_from_file(path_to_file)
            except Exception as exception:
                LOGGER.warning("There is a problem while computing mfcc on: " + path_to_file)
                continue

            features = np.hstack((features, features_file)) if features != [] else features_file

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
    _, values_possible = model.learn_model(features)
    LOGGER.info("Done. Converged: " + str(model.dpgmm_model.converged_))

    return [model, values_possible, file_taken]

def main(args):
    if not os.path.exists(os.path.join(args.folder_out, "model")):
        os.makedirs(os.path.join(args.folder_out, "model"))

    feature_extractor = FeatureMfcc(windows=0.06, shift=0.03,
                                    freq_min=args.freq_min,
                                    freq_max=args.freq_max,
                                    n_mfcc=26,
                                    energy=True)

    model, values_possible, file_taken = learn_model(args.folder_audio,
                                                     feature_extractor,
                                                     args.max_learn)

    save_obj(model, os.path.join(args.folder_out, 'model/model'))
    save_obj(values_possible, os.path.join(args.folder_out, 'model/values_possible'))
    save_obj(feature_extractor, os.path.join(args.folder_out, 'model/feature_extractor'))
    file_out_taken = open(os.path.join(args.folder_out, 'model/file_taken.txt'), 'w')
    for item in file_taken:
        file_out_taken.write("%s\n" % item)

    # TODO compute indices performances
#    plot_internal_indices.plot_silhouette(silhouette_score)


if __name__ == '__main__':
    # prepare parser of arguments
    PARSER = argparse.ArgumentParser(
        description='Process the algorithm to segment the audio and learn a model')
    PARSER.add_argument('folder_audio', metavar='folder_audio_in', type=str,
                        help='folder containing the audio files')
    PARSER.add_argument('folder_out', metavar='folder_out', type=str,
                        help='folder containing the results')
    PARSER.add_argument('freq_min', metavar='min_frequency', type=int,
                        help='minimum frequency')
    PARSER.add_argument('freq_max', metavar='max_frequency', type=int,
                        help='maximum frequency')
    PARSER.add_argument('-ml', '--max_learn', type=float,
                        help='between 0 and 1, help to define the number of files' +
                        ' use in every folder. 1 means every file is used.\n' +
                        ' By default all of them',
                        default=None)
    PARSER.add_argument('-v', '--verbose', action='store_true', help='Show every log')
    PARSER.add_argument('-l', '--logFile', type=str,
                        help='File where the logs will be saved', default=None)

    # parse arguments
    ARGS = PARSER.parse_args()

    # configure logging
    with open('config/logging.json') as config_description:
        config_log = ast.literal_eval(config_description.read())
        logging.config.dictConfig(config_log)

    LOGGER = logging.getLogger('activityDetectorDefault')
    if ARGS.verbose:
        LOGGER.setLevel(logging.DEBUG)

    if ARGS.logFile:
        # TODO correct: the logs in the file should be the same as activityDetectorDefault
        LOGGER.addHandler(logging.handlers.RotatingFileHandler(ARGS.logFile))

    main(ARGS)
