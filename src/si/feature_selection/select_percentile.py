from si.data.dataset import Dataset
import numpy as np
from typing import Callable

class SelectPercentile:

    def __int__(self, score_fun, percentile):
        self.score_func = score_fun
        self.percentile = percentile

        self.F =None
        self.p= None

    def fit(self, dataset):

        self.F, self.p = self.score_func(dataset)
        return self

    def transform(self, dataset):
        index = np.argsort(self.F)[::-1]  # ordem decrescente
        sort_vals = np.sort(self.F)[::-1]
        perc_vals = np.percentile(sort_vals, self.percentile)

        index = index[:sum(sort_vals <= perc_vals)]
        if dataset.features:
            features = np.array(dataset.features)[index]
        else:
            features = None

        return Dataset(dataset.X[:, index], dataset.Y, features, dataset.Label)

    def fit_transform(self, dataset):
        model = self.fit(dataset)
        return model.transform(dataset)
