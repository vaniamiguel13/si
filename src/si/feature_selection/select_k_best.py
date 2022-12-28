from typing import Callable
import numpy as np
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification

class SelectKBest:

    """
    Filtragem da dados baseada em F-scores. Seleciona apenas as melhores variáveis (k).
    """

    def __init__(self, score_func, k: int = 10):
        """
        Select features according to the k highest scores.
        Parameters
        ----------
        score_func: callable
            Function taking dataset and returning a pair of arrays (scores, p_values)
        k: int, default=10
            Number of top features to select.
        """
        self.score_func = score_func
        self.k = k

        self.F = None
        self.p = None

    def fit(self, dataset):
        """
        It fits SelectKBest to compute the F scores and p-values.
        Parameters
        ----------
        dataset: Dataset
            A labeled dataset
        """
        self.F, self.p = self.score_func(dataset)
        return self

    def transform(self, dataset):
        """
        It transforms the dataset by selecting the k highest scoring features.
        Parameters
        ----------
        dataset: Dataset
            A labeled dataset
        """

        index = np.argsort(self.F)[-self.k:][::-1]  # ordem decrescente
        features = np.array(dataset.Features)[index]

        return Dataset(dataset.X[:, index], dataset.Y, features, dataset.Label)

    def fit_transform(self, dataset):
        """
        It fits SelectKBest and transforms the dataset by selecting the k highest scoring features.
        Parameters
        ----------
        dataset: Dataset
            A labeled dataset
        """
        model = self.fit(dataset)
        return model.transform(dataset)





