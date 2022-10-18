from si.data.dataset import Dataset
import numpy as np
from typing import Callable

class SelectKBest:

    '''
    Filtragem da dados baseada em F-scores. Seleciona apenas as melhores variáveis (k).
    '''

    def __int__(self, score_func, k:int):
        self.score_func = score_func
        self.k = k

        self.F= None
        self.p= None

    def fit(self, dataset):
        self.F, self.p = self.score_func(dataset)
        return self

    def transform(self, dataset):

        index = np.argsort(self.F)[-self.k:][::-1]  # ordem decrescente
        features = np.array(dataset.features)[index]

        return Dataset(dataset.X[:, index], dataset.Y, features, dataset.Label)

    def fit_transform(self, dataset):
        model = self.fit(dataset)
        return model.transform(dataset)





