import numpy as np
from si.data.dataset import Dataset


class VarianceThreshold:
    """
    Variance Threshold feature selection.
    Features with a training-set variance lower than this threshold will be removed from the dataset.
    """

    def __init__(self, threshold):
        """
        Variance Threshold feature selection.
        Features with a training-set variance lower than this threshold will be removed from the dataset.
        Parameters
        ----------
        threshold: float
            The threshold value to use for feature selection. Features with a
            training-set variance lower than this threshold will be removed.
        """
        if threshold < 0:
            raise ValueError("Threshold must be non-negative")

        self.threshold = threshold
        self.variance = None

    def fit(self, dataset):
        """
        Fit the VarianceThreshold model according to the given training data.
        Parameters
        ----------
        dataset : Dataset
            The dataset to fit.
        """
        self.variance = np.var(dataset.X, axis=0)
        return self

    def transform(self, dataset):
        """
        It removes all features whose variance does not meet the threshold.
        Parameters
        ----------
        dataset: Dataset
        """
        mask = self.variance > self.threshold
        X = dataset.X
        X = X[:, mask]
        features = np.array(dataset.Features)[mask]

        if not (dataset.Features is None):
            dataset.Features = [elem for ix, elem in enumerate(dataset.Features) if mask[ix]]

        return Dataset(X, dataset.Y, list(features), dataset.Label)

    def fit_transform(self, dataset):
        """
        Fit to data, then transform it.
        Parameters
        ----------
        dataset: Dataset
        """
        Model = self.fit(dataset)
        return Model.transform(dataset)


if __name__ == "__main__":
    import numpy as np

    dataset = Dataset(np.array([[0, 2, 0, 3],
                                [1, 4, 2, 5],
                                [1, 2, 0, 1],
                                [0, 3, 0, 2]]),
                      np.array([1, 2, 3, 4]),
                      ["1", "2", "3", "4"], "5")

    temp = VarianceThreshold(1)
    print(temp.fit_transform(dataset))
