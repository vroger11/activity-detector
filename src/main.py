import src.compute_clusters.gaussian_mixtures as gausian_mixtures
import src.compute_features.mfcc as mfcc
import src.plotting_clusters.plot_informations as plt_clusters
import os
import numpy as np
import librosa
from scipy.stats.mstats import zscore
import argparse

# parse arguments
parser = argparse.ArgumentParser(description='Process the algorithm to segment the audio')
parser.add_argument('folder_audio', metavar='folder_audio_in', type=str,
                    help='folder containing the audio files')

args = parser.parse_args()
files = os.listdir(args.folder_audio)

##get features
print("Getting features")
features = []
for file in files:
    print("Getting features from: " + file)
    features_file = mfcc.get_mfcc_from_file(args.folder_audio + file, windows=0.06, shift=0.03, freq_min=1500, freq_max=8000)
    features = np.hstack((features, features_file)) if features != [] else features_file

##normalisation
# features = zscore(features, axis=0, ddof=1) seems useless
features = np.transpose(
    features)  # model require n*d; with n the number of observations and d the dimension of an observation

##learn model
print("Learning model")
model = gausian_mixtures.Model(n_components=30, n_iter=100, alpha=.5, verbose=0, covariance_type="diag")
model.learn_model(features)
print("Done. Converged: ", model.dpgmm_model.converged_)

## plot result
print("Plotting results")
clusters = model.predic_clusters(features)
print(np.unique(clusters))

signal = []
fs = 0
for file in files:
    signal_file, fs = librosa.load(args.folder_audio + file)
    signal = np.hstack((signal, signal_file)) if signal != [] else signal_file

m_clusters = plt_clusters.vector_of_cluster_to_matrix(clusters) #, number_max=model.dpgmm_model.n_components)
plt_clusters.show_audio_with_cluster(signal, fs, m_clusters, show_signal=False)
