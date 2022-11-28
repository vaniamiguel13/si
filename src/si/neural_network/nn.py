from si.data.dataset import Dataset

class NN:

    def __init__(self, layers=None):
        if layers is None:
            layers = []
        self.layers = layers

    def fit(self, dataset: Dataset) -> 'NN':

        data = dataset.X
        for layer in self.layers:
            data = layer.forward(data)
        return data

