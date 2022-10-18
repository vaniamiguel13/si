import numpy as np


def train_test_split(dataset, test_size, random_state):
    np.random.seed(random_state)

    n_samples= dataset.shape()[0]

    n_test= int(n_samples) * test_size

    permutations = np.rando.permutation(n_samples)

    test_indx= permutations[:n_test]

    train_indx= permutations[n_test:]

    train = Dataset(dataset.X[train_indx], dataset.y[train_indx], features = dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_indx], dataset.y[test_indx], features = dataset.features, label=dataset.label)
    return train, test
