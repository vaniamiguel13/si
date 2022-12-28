import numpy as np


def accuracy(y_true, Y_pred):
    """
    It returns the accuracy of the model on the given dataset
    Parameters
    ----------
    y_true: np.ndarray
        The true labels of the dataset
    y_pred: np.ndarray
        The predicted labels of the dataset
    """
    # error= (VN + VP) / (VN + VP + FP + FN)
    error = np.sum(y_true == Y_pred) / len(y_true)
    return error
