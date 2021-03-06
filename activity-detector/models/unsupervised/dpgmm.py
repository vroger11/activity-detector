"""
    Module computing Dirichlet Process Gaussian Mixture Model
"""

import sys
import logging.config
from sklearn import mixture
import numpy as np
from .abs_model import AbsModel


LOGGER = logging.getLogger('activityDetectorDefault')


class DPGMM(AbsModel):
    """
        class to learn and predict from observation with a DPGMM method

        The data must be n*d, with n the number of observations and d
        the dimension of an observation
    """

    def __init__(self, n_components=5, covariance_type='diag', n_iter=100, alpha=1.0, verbose=0):
        '''

            :param max_cluster:
            :param covariance_type:
        '''
        self.dpgmm_model = mixture.DPGMM(n_components=n_components, covariance_type=covariance_type,
                                         alpha=alpha, random_state=None, thresh=None, tol=0.001,
                                         verbose=verbose, min_covar=None, n_iter=n_iter,
                                         params='wmc', init_params='wmc')
        self.number_of_cluster_found = -1  # because the model was not learned

    def learn_and_fit(self, data_in):
        """
            learn a model fitting data_in

            :param data_in: observations n*d

            :return: clusters founded
        """

        return self.dpgmm_model.fit_predict(data_in)

    def learn_model(self, data_in):
        '''

            :param data_in: observations n*d
            :return:
        '''

        predicted = self.dpgmm_model.fit_predict(data_in)
        values_possible = np.unique(predicted)
        return [predicted, values_possible]

    def predic_clusters(self, data_in):
        '''
            Predicts cluster

            :param data_in: observations n*d
            :return: vector of cluster found
        '''

        if self.dpgmm_model.converged_:
            return self.dpgmm_model.predict(data_in)
        else:
            LOGGER.warning("The model does not converged", file="stderr")
            sys.exit(1)

    def generate_data_from_model(self, number_of_data=1000):
        '''
            Generate data from model learned

            :param number_of_data: number of data to generate
            :return:
        '''
        if self.dpgmm_model.converged_:
            return self.dpgmm_model.sample(number_of_data)
        else:
            LOGGER.warning("The model does not converged", file="stderr")
            sys.exit(2)
