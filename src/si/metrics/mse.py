import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    It returns the mean squared error of the model on the given dataset

    Parameters
    ----------
    y_true: np.ndarray
        The true labels of the dataset
    y_pred: np.ndarray
        The predicted labels of the dataset

    Returns
    -------
    mse: float
        The mean squared error of the model
    """
    return np.sum((y_true - y_pred) ** 2) / (y_true.shape[0] * 2)


def mse_derivate(y_true: np.ndarray, y_pred: np.ndarray) -> float:

    N = y_true.shape[0]
    return - (y_true - y_pred) / N
