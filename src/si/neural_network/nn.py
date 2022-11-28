from si.data.dataset import Dataset
import numpy as np
from si.metrics import mse
from typing import Callable

class NN:

    def __init__(self, layers=None, epochs = 1000,learning_rate: float = 0.01, loss_fun: Callable = mse,
                 loss_derivate: Callable = mse_derivate, verbose: bool = False ):
        if layers is None:
            layers = []
        self.layers = layers
        self.epochs = epochs
        self.loss_fun = loss_fun
        self.learning_rate = learning_rate
        self.loss_derivate = loss_derivate
        self.verbose = verbose

        self.history={}

    def fit(self, dataset: Dataset) -> 'NN':

        X = dataset.X
        Y = dataset.Y

        for epoch in range(1,  self.epochs + 1):

            for layer in self.layers:
                X = layer.forward(X)

            error = self.loss_derivate(Y,X)
            for layer in self.layers[::-1]:
                error= layer.backward(error, self.learning_rate)

            cost= self.loss_fun(Y, X)
            self.history[epoch]=cost

            if self.verbose:
                print(f'Epoch {epoch}/{self.epochs} - cost: {cost}')

        return self

    def predict(self, dataset: Dataset) -> np.ndarray:

        X = dataset.X

        for layer in self.layers:
            X = layer.forward(X)

        return X

    def cost(self, ):

