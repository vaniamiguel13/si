import numpy as np

def accuracy(y_true, Y_pred):

    #error= (VN + VP) / (VN + VP + FP + FN)
    error = np.sum(y_true == Y_pred) / len(y_true)
    return error