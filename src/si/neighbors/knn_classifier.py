from si.statistics.euclidean_distance import euclidean_distance
from typing import Callable, Union
import numpy as np
from si.metrics.accuracy import accuracy
from si.data.dataset import Dataset


class KNNClassifier:

    def __init__(self, k: int, distance: Callable = euclidean_distance):
        self.dataset = None
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


if __name__ == '__main__':
    # import dataset
    from si.data.dataset import Dataset
    from si.model_selection.split import train_test_split

    # load and split the dataset
    dataset_ = Dataset.from_random(600, 100, 2)
    dataset_train, dataset_test = train_test_split(dataset_, test_size=0.2)

    # initialize the KNN classifier
    knn = KNNClassifier(k=3)

    # fit the model to the train dataset
    knn.fit(dataset_train)

    # evaluate the model on the test dataset
    score = knn.score(dataset_test)
    print(f'The accuracy of the model is: {score}')
