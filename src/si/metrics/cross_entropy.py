import numpy as np


def cross_entropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred)) / len(y_true)


def cross_entropy_derivate(y_true, y_pred):
    return -y_true / (len(y_true)*y_pred)
