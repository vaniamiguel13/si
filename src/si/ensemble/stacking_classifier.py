from si.data.dataset import Dataset
import numpy as np
from si.metrics.accuracy import accuracy


class StackingClassifier:

    def __init__(self, models: list, final_mod):
        self.models = models  # lista de modelos já inicializados
        self.final_mod = final_mod

    def fit(self, dataset):
        dt = Dataset(dataset.X, dataset.Y, dataset.Features, dataset.Label)
        for model in self.models:
            model.fit(dataset)
            dt.X = np.c_[dt, model.predict(dataset)]

        self.final_mod.fit(dt)
        return self

    def predict(self, dataset):
        dt = Dataset(dataset.X, dataset.Y, dataset.Features, dataset.Label)
        for model in self.models:
            dt.X = np.c_[dt.X, model.predict(dataset)]

        return self.final_mod.predict(dt)

    def score(self, dataset):
        y_pred_ = self.predict(dataset)
        return accuracy(dataset.Y, y_pred_)
