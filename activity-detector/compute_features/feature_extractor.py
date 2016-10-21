import logging
from mfcc import FeatureMfcc


def configure_feature_extractor(feature_description):
    """
        :param feature_description: dictionnary containing feature description
        :return: feature extractor

        configure a feature extractor corresponding to feature_description
    """

    if feature_description['name'] == 'mfcc':
        return FeatureMfcc(windows=feature_description['window'],
                           shift=feature_description['shift'],
                           freq_min=feature_description['freq_min'],
                           freq_max=feature_description['freq_max'],
                           n_mfcc=feature_description['n_mfcc'],
                           energy=feature_description['energy'])
    else:
        LOGGER.warning("The feature is not recognized.")

LOGGER = logging.getLogger('activityDetectorDefault')
