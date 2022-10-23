import numpy as np
from si.metrics.rmse import rmse
from si.statistics.euclidean_distance import euclidean_distance

class KNNRegressor:

    def __int__(self, k: int , distance = euclidean_distance):
        self.k = k
        self.distance = distance


    def fit(self, dataset):
        self.dataset = dataset
        return self

    def predict(self, dataset):

        return np.apply_along_axis(self._get_closest_label, axis=1, arr=dataset.X)

    def _get_closest_label(self, sample: np.ndarray):

        distances = self.distance(sample, self.dataset.X)

        # get the k nearest neighbors
        indexes = np.argsort(distances)[:self.k]

        # get the labels of the closet neighbors
        closest_y = self.dataset.y[indexes]

        # get mean
        media = np.mean(closest_y)
        return media


    def score(self, dataset):
        return rmse(dataset.y, self.predict(dataset))

