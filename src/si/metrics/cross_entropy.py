import numpy as np


def cross_entropy(y_true, y_pred):
    """
    Cross entropy for a given model
    """
    return -np.sum(y_true * np.log(y_pred)) / len(y_true)


def cross_entropy_derivate(y_true, y_pred):
    """
    Cross entropy derivative for a given model
    """
    return -y_true / (len(y_true)*y_pred)
