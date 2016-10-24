"""
    Provide a function to initialize any feature extractor from a dictionnary
"""
import logging
from .unsupervised.dpgmm import DPGMM


def configure_model(model_description):
    """
        :param feature_description: dictionnary containing model description
        :return: model initialised

        configure a model corresponding to model_description
    """

    if model_description['name'] == 'DPGMM':
        return DPGMM(n_components=model_description['n_components'],
                     n_iter=model_description['n_iter'],
                     alpha=model_description['alpha'],
                     verbose=0,
                     covariance_type=model_description['covariance_type'])
    else:
        LOGGER.warning("The model is not recognized.")

LOGGER = logging.getLogger('activityDetectorDefault')
