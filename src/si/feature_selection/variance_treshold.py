import numpy as np
from si.data.dataset import Dataset

class VarianceThreshold:
    def __init__(self, treshold):
        self.threshold = treshold
        self.variance = None

    def fit(self, dataset):

        self.variance = dataset.get_var()
        return self

    def transform(self, dataset):
        mask = self.variance > self.threshold
        new_X = dataset.X[:, mask]

        if not (dataset.Features is None):
            dataset.Features = [elem for ix, elem in enumerate(dataset.Features) if mask[ix]]

        return Dataset(new_X, dataset.Y, dataset.Features, dataset.Label)

    def fit_transform(self, dataset):
        model = self.fit(dataset)
        return model.transform(dataset)

if __name__ == '__main__':
    from si.io.CSV import read_csv
    # make a linear dataset
    data0 = read_csv('D:/Mestrado/2ano/1semestre/SIB/si/datasets/cpu/cpu.csv', sep=",", label=True)

    # fit the model
    model = VarianceThreshold(3)
    print(data0.X)
    fit=(model.fit(data0))
    print(model.transform(fit))

