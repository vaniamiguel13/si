import numpy as np
from si.data.dataset import Dataset
from sklearn import preprocessing

class PCA:
    """
    Principal Component Analysis (PCA), using the Singular Value Decomposition (SVD) method.
    """

    def __init__(self, n_components:int):
        """
        Guarda as variáveis

        :param n_components: Número de componentes a serem retornados após análise
        """
        self.n_components= n_components

    def fit(self, dataset: Dataset):
        """
        Armazena os valores médios de cada amostra, o primeiro n componentes principais (especificados pelo usuário)
        e as respetivas variâncias explicadas.

        :param dataset: isntância da classe Dataset
        """
        self.mean= np.mean(dataset.X, axis=0)

        self.cent_data = np.subtract(dataset.X, self.mean)

        # X = U*S*VT
        U, S, V_t = np.linalg.svd(self.cent_data, full_matrices=False)

        self.comp_princ = V_t[:self.n_components]

        #EV = S^2/(n-1) – n
        n = len(dataset.X[:, 0])
        EV= (S**2)/(n-1)-n
        self.explained_variance = EV[:self.n_components]

        return self

    def transform(self, dataset: Dataset):
        """
        Retorna o SVD reduzido
        :param dataset: isntância da classe Dataset
        """

        V = self.comp_princ.T # matriz transporta
        # SVD reduced
        Xreduced = np.dot(self.cent_data, V)

        return Xreduced



