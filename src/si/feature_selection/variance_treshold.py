import numpy as np
from si.data.dataset import Dataset

class VarianceThreshold:
    def __int__(self, treshold):
        self.threshold = treshold
        self.variance = None

    def fit(self, dataset):

        self.variance= dataset.get_var()
        return self

    def transform(self, dataset):
        mask = self.variance > self.threshold
        new_X = dataset.X[:, mask]

        if not (dataset.features is None):
            dataset.features = [elem for ix, elem in enumerate(dataset.features) if mask[ix]]

        return Dataset(new_X, dataset.y, dataset.features, dataset.label)

    def fit_transform(self, dataset):
        model = self.fit(dataset)
        return model.transform(dataset)


