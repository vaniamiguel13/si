from si.data.dataset import Dataset
from si.io.CSV import read_csv
import itertools
import numpy as np


class KMer:
    """
    A sequence descriptor that returns the k-mer composition of the sequence.

    Parameters
    ----------
    k : int
        The k-mer length.

    Attributes
    ----------
    k_mers : list of str
        The k-mers.
    """

    def __init__(self, k: int = 3, alphabet: str = "DNA"):
        """
        Parameters
        ----------
        :param k : int
            The k-mer length.
        """
        # check parameter values
        if k < 1:
            raise ValueError("The value of 'k' must be a positive integer.")
        if alphabet not in ["DNA", "PROT"]:
            raise ValueError("The value of 'alphabet' must be in {'DNA', 'PROT'}.")
        # parameters
        self.k = k
        # attributes
        if alphabet.upper() == 'DNA':
            self.alph = 'ACTG'
        elif alphabet.upper() == 'PROT':
            self.alph = 'ACDEFGHIKLMNPQRSTVWY'
        self.k_mers = None

    def fit(self, dataset: Dataset) -> "KMer":
        """
        Fits the descriptor to the dataset.

        Parameters
        ----------
        dataset : Dataset
            The dataset to fit the descriptor to.

        Returns
        -------
        KMer
            The fitted descriptor.
        """
        # generate the k-mers
        self.k_mers = [''.join(k_mer) for k_mer in itertools.product(self.alph, repeat=self.k)]
        return self

    def _get_sequence_k_mer_composition(self, sequence: str) -> np.ndarray:
        """
        Calculates the k-mer composition of the sequence.

        Parameters
        ----------
        sequence : str
            The sequence to calculate the k-mer composition for.

        Returns
        -------
        list of float
            The k-mer composition of the sequence.
        """
        # calculate the k-mer composition
        counts = {k_mer: 0 for k_mer in self.k_mers}

        for i in range(len(sequence) - self.k + 1):
            k_mer = sequence[i:i + self.k]
            counts[k_mer] += 1

        return np.array([counts[k_mer] / len(sequence) for k_mer in self.k_mers])  # normalizar as contagens

    def transform(self, dataset: Dataset)-> Dataset:
        """
        Transforms the dataset.

        Parameters
        ----------
        dataset : Dataset
            The dataset to transform.

        Returns
        -------
        Dataset
            The transformed dataset.
        """
        # calculate the k-mer composition
        sequences_k_mer_composition = [self._get_sequence_k_mer_composition(sequence) for sequence in dataset.X[:, 0]]
        sequences_k_mer_composition = np.array(sequences_k_mer_composition)

        return Dataset(X=sequences_k_mer_composition, y=dataset.Y, features=self.k_mers, label=dataset.Label)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        Fits the descriptor to the dataset and transforms the dataset.

        Parameters
        ----------
        dataset : Dataset
            The dataset to fit the descriptor to and transform.

        Returns
        -------
        Dataset
            The transformed dataset.
        """
        return self.fit(dataset).transform(dataset)
