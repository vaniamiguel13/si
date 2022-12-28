from si.data.dataset import Dataset
import numpy as np
from typing import Callable
from si.statistics.f_classification import f_classification


class SelectPercentile:
    """
    Class that filters dataset variables based on their F-scores. Selects all variables
    with F-score values above the specified corresponding percentile.
    """

    def __init__(self, score_fun: Callable[[object], tuple], percentile):
        """
        Stores the input values.

        Paramaters
        ----------
        :param score_func: f_classification() or f_regression() functions.
        :param percentile: Percentile value cut-off. Only F-scores above this
                           value will remain in the filtered dataset.
        """
        self.score_func = score_fun
        self.percentile = percentile

        self.F = None
        self.p = None

    def fit(self, dataset):
        """
        Stores the F-scores and respective p-values of each variable of the given dataset.

        Paramaters
        ----------
        :param dataset: An instance of the Dataset class.
        """

        self.F, self.p = self.score_func(dataset)
        return self

    def transform(self, dataset):
        """
        Returns a filtered version of the given Dataset instance using their
        F-scores. The new dataset will have only the variables with F-scores above
        the specified percentile value.

        Paramaters
        ----------
        :param dataset: An instance of the Dataset class.
        """
        index = np.argsort(self.F)[::-1]  # ordem decrescente
        sort_vals = np.sort(self.F)[::-1]
        perc_vals = np.percentile(sort_vals, self.percentile)

        index = index[:sum(sort_vals <= perc_vals)]
        if dataset.Features:
            features = np.array(dataset.Features)[index]
        else:
            features = None

        return Dataset(dataset.X[:, index], dataset.Y, features, dataset.Label)

    def fit_transform(self, dataset):
        """
        Calls the fit() and transform() methods, returning the filtered version
        of the given Dataset instance.

        Paramaters
        ----------
        :param dataset: An instance of the Dataset class.
        """
        model = self.fit(dataset)
        return model.transform(dataset)


if __name__ == "__main__":
    from si.io.CSV import read_csv
    X = np.array([[1, 2, 3, 4], [3, 6, 5, 1], [7, 4, 1, 5], [1, 3, 2, 9]])
    y = np.array([1, 1, 0, 0])
    ds = Dataset(X, y)
    ds = read_csv('D:/Mestrado/2ano/1semestre/SIB/si/datasets/cpu/cpu.csv', sep=",", label=True)
    selector = SelectPercentile(f_classification, percentile=0.4)
    new_ds = selector.fit_transform(ds)
    print(new_ds.X)
