import numpy as np
def sigmoid_function(X):

    prob = 1/(1 + (np.e)**(-X))
    return prob

# print(sigmoid_function(np.array([1, 2, 3])))