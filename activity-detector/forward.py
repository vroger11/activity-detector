"""
    Use to forward a model previously learned
"""

import os
import argparse
import logging.config
import ast
import json
import numpy as np
from scipy.stats import mstats
from tools.manipulate_objects import load_obj


def forward_model(folder_out, folder_audio, model, feature_extractor):
    if not os.path.exists(os.path.join(folder_out, "clusters")):
        os.makedirs(os.path.join(folder_out, "clusters"))

    LOGGER.info("Saving results")
    for root, _, files in os.walk(folder_audio):
        LOGGER.info("Saving in: " + folder_out)
        for file in files:
            path_to_file = os.path.join(root, file)
            try:
                features_file = feature_extractor.get_feature_from_file(path_to_file)
            except Exception as exception:
                LOGGER.warning("There is a problem with: " + path_to_file)
                LOGGER.warning(exception)
                continue

            features = mstats.zscore(features_file, axis=1, ddof=1)
            features = np.transpose(features)
            clusters = model.predic_clusters(features)

            # save results
            file_out, _ = os.path.splitext(file)
            path_out_forwarded = os.path.join(folder_out, "clusters/" + root.replace(folder_audio, ''))
            if not os.path.exists(path_out_forwarded):
                os.makedirs(path_out_forwarded)

            path_out_forwarded = os.path.join(path_out_forwarded, file_out + ".txt")
            np.savetxt(path_out_forwarded, clusters, delimiter=" ", fmt='%i',)

def main(args):
    # get model
    model = load_obj(os.path.join(args.folder_model, 'model'))
    feature_extractor = load_obj(os.path.join(args.folder_model, 'feature_extractor'))

    if not os.path.exists(args.folder_out):
        os.makedirs(args.folder_out)

    dic_forward = dict()
    dic_forward['model_used'] = args.folder_model
    dic_forward['data_forwarded'] = args.folder_audio
    with open(os.path.join(args.folder_out, 'forward_info.json'), 'w') as fp:
        json.dump(dic_forward, fp)

    # plot result
    forward_model(args.folder_out,
                  args.folder_audio,
                  model,
                  feature_extractor)

if __name__ == '__main__':
    # prepare parser of arguments
    PARSER = argparse.ArgumentParser(description='Process the algorithm to segment the audio')
    PARSER.add_argument('folder_audio', metavar='folder_audio_in', type=str,
                        help='folder containing the audio files')
    PARSER.add_argument('folder_model', metavar='folder_model', type=str,
                        help='folder containing model + feature extractor')
    PARSER.add_argument('folder_out', metavar='folder_out', type=str,
                        help='folder containing the results')
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
