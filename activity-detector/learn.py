"""
    Use to learn a model
"""

import os
import argparse
import logging.config
import ast
import random
import numpy as np
from scipy.stats import mstats
from models.config_model import configure_model
from compute_features.config_extractor import configure_feature_extractor
from tools.manipulate_objects import save_obj


def learn_model(model, folder_audio, feature_extractor, max_learn):
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

        if not max_learn is None:
            number_of_file = int(len(files)*max_learn)
            random.shuffle(files)
            files_to_learn = files[0:number_of_file]
        else:
            files_to_learn = files

        file_taken.append(files_to_learn)
        for file in files_to_learn:
            path_to_file = os.path.join(root, file)

            try:
                features_file = feature_extractor.get_feature_from_file(path_to_file)
            except Exception:
                LOGGER.warning("There is a problem while computing mfcc on: " +
                               path_to_file)
                continue

            features = np.hstack((features, features_file)) if features != [] else features_file

    features = mstats.zscore(features, axis=1, ddof=1)
    # model require n*d; with n the number of observations and d the dimension of an observation
    features = np.transpose(features)

    # learn model
    LOGGER.info("Learning model")
    _, values_possible = model.learn_model(features)
    LOGGER.info("Done. Converged: " + str(model.dpgmm_model.converged_))

    return [values_possible, file_taken]

def main(args, experience_description):
    """
        basic things to do when learning a model
    """

    if not os.path.exists(os.path.join(args.folder_out, "model")):
        os.makedirs(os.path.join(args.folder_out, "model"))
    else:
        LOGGER.warning('Directory ' + os.path.join(args.folder_out, "model") +
                       "already exists")

    # prepare feature extractor
    feature_extractor = configure_feature_extractor(experience_description['feature'])
    #prepare the model
    model = configure_model(experience_description['model'])

    values_possible, file_taken = learn_model(model,
                                              args.folder_audio,
                                              feature_extractor,
                                              args.max_learn)

    save_obj(model, os.path.join(args.folder_out, 'model/model'))
    save_obj(values_possible, os.path.join(args.folder_out, 'model/values_possible'))
    save_obj(feature_extractor, os.path.join(args.folder_out, 'model/feature_extractor'))
    file_out_taken = open(os.path.join(args.folder_out, 'model/file_taken.txt'), 'w')
    for item in file_taken:
        file_out_taken.write("%s\n" % item)


if __name__ == '__main__':
    # prepare parser of arguments
    PARSER = argparse.ArgumentParser(
        description='Process the algorithm to segment the audio and learn a model')
    PARSER.add_argument('folder_audio', metavar='folder_audio_in', type=str,
                        help='folder containing the audio files')
    PARSER.add_argument('folder_out', metavar='folder_out', type=str,
                        help='folder containing the results')
    PARSER.add_argument('config', metavar='config', type=str,
                        help='configuration file of the feature, model, ... used. ' +
                        'See config/example.json for an example.')
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
        CONFIG_LOG = ast.literal_eval(config_description.read())
        logging.config.dictConfig(CONFIG_LOG)

    with open(ARGS.config) as config_description:
        CONFIG_EXPERIENCE = ast.literal_eval(config_description.read())

    LOGGER = logging.getLogger('activityDetectorDefault')
    if ARGS.verbose:
        LOGGER.setLevel(logging.DEBUG)

    if ARGS.logFile:
        # TODO correct: the logs in the file should be the same as activityDetectorDefault
        LOGGER.addHandler(logging.handlers.RotatingFileHandler(ARGS.logFile))

    main(ARGS, CONFIG_EXPERIENCE)
