import numpy as np
import math

def rmse(y_true, Y_pred):

    error = np.square(np.subtract(y_true,Y_pred)).mean()
    serror = math.sqrt(error)
    return serror
