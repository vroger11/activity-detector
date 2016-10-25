"""
    Module defining function to save/load python objects (models, feature extractor, ...)
"""

import pickle

def load_obj(filename):
    """
        :param filename: where the object is saved
    """
    with open(filename + '.pkl', 'rb') as file:
        return pickle.load(file)

def save_obj(obj, filename):
    """
        :param obj: object to save in a pickle file
        :param filename: where the object will be saved
    """

    with open(filename + '.pkl', 'wb') as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)
