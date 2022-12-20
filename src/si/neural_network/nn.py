from si.data.dataset import Dataset
import numpy as np
from si.metrics.mse import mse, mse_derivate
from typing import Callable


class NN:

    def __init__(self, layers=None, epochs = 1000,learning_rate: float = 0.01, loss_fun: Callable = mse,
                 loss_derivate: Callable = mse_derivate, verbose: bool = False ):
        if layers is None:
            layers = []
        self.layers = layers
        self.epochs = int(epochs)
        assert self.epochs >= 1, "epochs >= 1"
        self.loss_fun = loss_fun
        self.learning_rate = learning_rate
        self.loss_derivate = loss_derivate
        self.verbose = verbose

        self.history={}

    def fit(self, dataset: Dataset) -> 'NN':

        for epoch in range(1,  self.epochs + 1):

            X = np.array(dataset.X)
            Y = np.reshape(dataset.Y, (-1, 1))

            # forward propagation
            for layer in self.layers:
                X = layer.forward(X)

            #backward propagation
            error = self.loss_derivate(Y,X)
            for layer in self.layers[::-1]:
                error= layer.backward(error, self.learning_rate)

            #save history
            cost = self.loss_fun(Y, X)
            self.history[epoch] = cost

            if self.verbose:
                print(f'Epoch {epoch}/{self.epochs} - cost: {cost}')

        return self

    def predict(self, dataset: Dataset) -> np.ndarray:

        X = dataset.X

        for layer in self.layers:
            X = layer.forward(X)

        return X


