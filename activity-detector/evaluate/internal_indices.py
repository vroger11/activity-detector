"""
    Module containing different internal indices methods
"""
import numpy as np
from sklearn import metrics


def silhouette_clusters(data, clusters):
    """
        :param data: n*d where n is the number of observations and d the dimensions of
                     each observation
        :param clusters: an array of length n

        compute silhoutte score for every cluster
    """
    silhouette_samples_score = metrics.silhouette_samples(data, clusters, metric='euclidean')
    values_possible = np.unique(clusters)
    silhouette_mean_clusters = np.zeros((1, len(values_possible)))
    k = 0
    for i in values_possible:
        index = np.where(clusters == i)
        silhouette_mean_clusters[k] = np.mean(silhouette_samples_score[index])
        k += 1

    return silhouette_mean_clusters
