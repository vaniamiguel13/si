import numpy as np
from si.data.dataset import Dataset


def train_test_split(dataset, test_size, random_state):
    np.random.seed(random_state)

    n_samples= dataset.shape()[0]

    n_test= int(n_samples * test_size)

    permutations = np.random.permutation(n_samples)

    test_indx = permutations[:n_test]

    train_indx = permutations[n_test:]


    train = Dataset(dataset.X[train_indx], dataset.Y[train_indx], features = dataset.Features, label=dataset.Label)
    test = Dataset(dataset.X[test_indx], dataset.Y[test_indx], features = dataset.Features, label=dataset.Label)

    return train, test
