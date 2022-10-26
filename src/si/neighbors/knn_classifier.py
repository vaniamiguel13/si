from si.statistics.euclidean_distance import euclidean_distance
import numpy as np
from si.metrics.accuracy import accuracy
from si.data.dataset import Dataset


class KNNClassifier:

    def __init__(self, k: int, distance=euclidean_distance):
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
        closest_y = self.dataset.Y[indexes]

        # get most common label
        labels, counts = np.unique(closest_y, return_counts=True)
        return labels[np.argmax(counts)]

    def score(self, dataset):
        return accuracy(dataset.Y, self.predict(dataset))