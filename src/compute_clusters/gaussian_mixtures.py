import sys
from sklearn import mixture
import logging.config
import ast

with open('config/logging.json') as f:
    config = ast.literal_eval(f.read())
    logging.config.dictConfig(config)

logger = logging.getLogger('activityDetectorDefault')


class Model:
    def __init__(self, n_components=5, covariance_type='diag', n_iter=100, alpha=1.0, verbose=0):
        '''

        :param max_cluster:
        :param covariance_type:
        '''
        self.dpgmm_model = mixture.DPGMM(n_components=n_components, covariance_type=covariance_type, alpha=alpha,
                                         random_state=None, thresh=None, tol=0.001, verbose=verbose, min_covar=None,
                                         n_iter=n_iter,
                                         params='wmc', init_params='wmc')
        self.number_of_cluster_found = -1  # because the model was not learned

    def learn_and_fit(self, data_in):
        return self.dpgmm_model.fit_predict(data_in)

    def learn_model(self, data_in):
        '''

        :param data_in:
        :return:
        '''

        self.dpgmm_model.fit(data_in)

    def predic_clusters(self, data_in):
        '''
        Predicts cluster

        :param data_in:
        :return: vector of cluster found
        '''

        if self.dpgmm_model.converged_:
            return self.dpgmm_model.predict(data_in)
        else:
            logger.warning("The model does not converged", file="stderr")
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
            logger.warning("The model does not converged", file="stderr")
            sys.exit(2)
