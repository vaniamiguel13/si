import numpy as np
import math


def rmse(y_true, Y_pred):
    error = np.square(np.subtract(y_true, Y_pred)).mean()
    serror = math.sqrt(error)
    return serror


if __name__ == "__main__":
    true = np.array([0, 1, 1, 1, 0, 1])
    pred = np.array([1, 0, 1, 1, 0, 1])
    print(f"RMSE: {rmse(true, pred):.4f}")
