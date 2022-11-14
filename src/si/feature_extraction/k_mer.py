from si.data.dataset import Dataset
from si.io.CSV import read_csv
import itertools
import numpy as np


class KMer:
    def __init__(self, k: int, alphabet = 'DNA'):
        self.k_mers = None
        self.k = k
        if alphabet.upper() == 'DNA':
            self.alph = 'ACTG'
        elif alphabet.upper() == 'PROT':
            self.alph = 'ACDEFGHIKLMNPQRSTVWY'

    def fit(self, dataset):
        self.k_mers = [''.join(k_mer) for k_mer in itertools.product(self.alph, repeat=self.k)]

    def _get_sequence_k_mer_composition(self, sequence: str) -> np.array:
        counts = {k_mer: 0 for k_mer in self.k_mers}
        for i in range(len(sequence) - self.k + 1):
            k_mer = sequence[i:i + self.k]
            counts[k_mer] += 1

        return np.array([counts[k_mer] / len(sequence) for k_mer in self.k_mers])  # normalizar as contagens

    def transform(self, dataset: Dataset):
        sequences_k_mer_composition = [self._get_sequence_k_mer_composition(sequence) for sequence in dataset.X[:, 0]]
        sequences_k_mer_composition = np.array(sequences_k_mer_composition)

        # create a new dataset
        return Dataset(X=sequences_k_mer_composition, y=dataset.Y, features=self.k_mers, label=dataset.Label)

if __name__ == '__main__':
    data1 = read_csv("D:/Mestrado/2ano/1semestre/SIB/si/datasets/tfbs/tfbs.csv", ",", True, True)
    x = KMer(3)
    x.fit(data1)
    print(x.transform(data1))
