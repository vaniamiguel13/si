import numpy as np
from si.statistics.sigmoid_function import sigmoid_function


class Dense:

    def __init__(self, input_size, output_size):
        self.input_size = input_size  # Same size as attributes of input data
        self.output_size = output_size

        # Can be any initial values!
        self.X = None
        self.weights = np.random.randn(input_size,
                                       output_size) * 0.01  # (lines, columns) (normal distribution between 0 e 1)
        self.bias = np.zeros((1, self.output_size))

    def forward(self, X: np.array):  # Forward propagation
        """

        Parameters
        ----------
        :param X: Input data matrix (examples x attributes)
        """
        self.X = X
        return np.dot(X,
                      self.weights) + self.bias  # Multiplpies input data lines (examples) with weights columns,
        # sums values of each column, and adds bias line

    def backward(self, error: np.ndarray, learning_rate: float = 0.001) -> np.ndarray:
        self.weights -= learning_rate * np.dot(self.X.T, error)
        error_propagate = np.dot(error, self.weights.T)
        self.bias -= learning_rate * np.sum(error, axis=0)

        return error_propagate


class SigmoidActivation:

    def __init__(self):
        self.X = None

    def forward(self, input_data: np.array):
        self.X = input_data
        return sigmoid_function(input_data)

    def backward(self, error: np.ndarray, learning_rate: bool = 0.001) -> np.ndarray:
        sigmoid_derivative = 1 / (1 + np.exp(-self.X))
        sigmoid_derivative = sigmoid_derivative * (1 - sigmoid_derivative)

        error_propagate = error * sigmoid_derivative

        return error_propagate


class SoftMaxActivation:
    def __init__(self):
        self.X = None

    def forward(self, input_data: np.array):
        self.X = input_data
        ez = np.exp(input_data)
        return ez / (np.sum(ez, axis=1, keepdims=True))

    def backward(self, error: np.ndarray, learning_rate_bool=0.001):
        return error


class ReLUActivation:

    def __init__(self):
        self.X = None

    def forward(self, input_data: np.array):
        self.X = input_data
        return np.maximum(input_data, 0)

    def backward(self, error: np.ndarray, learning_rate: bool = 0.001):
        error_propagate = np.where(self.X > 0, 1, 0)
        return error_propagate
