import numpy as np
import pandas as pd
from typing import Tuple, Sequence
from prettytable import PrettyTable


class Dataset:
    '''
    Cria um conjunto de dados e fornece informações de distribuição dos mesmos.
    '''

    def __init__(self, X, y=None, features=None, label=None):
        '''
        Guarda as Variáveis

        Parameters
        -----------
        :param X: Uma matriz de variável independente (deve ser uma instância numpy.ndarray).
        :param y: O vetor variável dependente (deve ser uma instância numpy.ndarray)
        :param features: Um vetor contendo os nomes de cada variável independente.
        :param label: O nome da variável dependente.
        '''

        # x = numy array
        # y = array de 1 dim
        # features = lista ou tuplo
        # label = str

        self.X = X
        self.Y = y
        self.Features = features
        self.Label = label

    def __str__(self):
        """
        Método para dar print do dataset
        """

        # 1º método -> sem packages
        # if not (self.Features is None):
        #     r = f'X:{str(self.Features)[:]}\n--\n'
        # else:
        #     r = 'X:\n--\n'
        # for elem in self.X:
        #     r += str(elem)[:].replace(' ', '\t') + '\n'
        # if not (self.Y is None):
        #     r += f'\nY: {self.Label}\n--\n' + str(self.Y).replace(' ', '\t') + '\n'
        # return r

        # 2º método -> com packages
        table = PrettyTable()
        if not (self.Features is None):
            table.field_names = self.Features
        for elem in self.X:
            table.add_row(elem)
        if not (self.Y is None) and not (self.Label is None):
            table.add_column(self.Label, self.Y)
        if not (self.Y is None) and (self.Label is None):
            table.add_column('y', self.Y)

        return str(table)

    def shape(self):
        '''
        Retorna as dimensões das variável independente.
        '''
        return self.X.shape

    def has_label(self):
        '''
        Verifica se a variável dependente está disponível.
        '''
        if self.Y is None:
            return False
        else:
            return True

    def get_classes(self):
        """
        Retorna os valores exclusivos da variável dependente.
        Caso não exista retorna uma mensagem de ERRO.
        """
        if self.Y is None:
            return 'ERRO'
        return np.unique(self.Y)

    def get_mean(self):
        '''
        Devolve a média de cada variável
        '''
        return np.mean(self.X, axis=0)

    def get_var(self):
        '''
        Devolve a variância de cada variável
        '''
        return np.var(self.X, axis=0)

    def get_median(self):
        '''
        Devolve a mediana de cada variável
        '''
        return np.median(self.X, axis=0)

    def get_min(self):
        '''
        Devolve o valor mínimo de cada variável
        '''
        return np.min(self.X, axis=0)

    def get_max(self):
        '''
        Devolve o valor máximo de cada variável
        '''
        return np.max(self.X, axis=0)

    def summary(self):
        '''
        Retorna pandas dataframe contendo a média, mediana, variância,
        valor mínimo e máximo de cada variável.
        '''
        return pd.DataFrame(
            {'mean': self.get_mean(),
             'var': self.get_var(),
             'median': self.get_median(),
             'min': self.get_min(),
             'max': self.get_max()})

    def drop_na(self):
        """
        Remove todas as linhas que contêm valores omissos.
        """

        df = pd.DataFrame(self.X, columns=self.Features)

        if not (self.Y is None):
            df.insert(loc=len(df), column=self.Label, value=self.Y)

            dfn = df.dropna()
            dt = dfn.to_numpy()

            if dt is not None:
                self.Y = []
                for elem in dt[0:, -1:]:
                    self.Y.append(float(elem))
                self.X = dt[0:, :-1]

        else:
            dfn = df.dropna()
            dt = dfn.to_numpy()
            self.Y = []
            self.X = dt[0:, :]

        return Dataset(self.X, self.Y, self.Features, self.Label)

    def fill_Na(self, n_or_m= 0):

        """
        Substitui os valores ausentes pelo valor de escolha do usuário.
        :param n_or_m: valor a substituir, 0 por default
        """

        df = pd.DataFrame(self.X, columns=self.Features)
        if not (self.Y is None):
            df.insert(loc=len(df), column=self.Label, value = self.Y)

            fill_df = df.fillna(n_or_m)
            dt = fill_df.to_numpy()

            if dt is not None:
                self.Y = []
                for elem in dt[0:, -1:]:
                    self.Y.append(float(elem))

                self.X = dt[0:, :-1]
        else:
            dfn = df.dropna()
            dt = dfn.to_numpy()
            self.Y = []
            self.X = dt[0:, :]

        return Dataset(self.X, self.Y, self.Features, self.Label)

    @classmethod
    # Cria um dataset random -> professor, usado para testes

    def from_random(cls,
                    n_samples: int,
                    n_features: int,
                    n_classes: int = 2,
                    features: Sequence[str] = None,
                    label: str = None):
        """
        Creates a Dataset object from random data
        Parameters
        ----------
        n_samples: int
            The number of samples
        n_features: int
            The number of features
        n_classes: int
            The number of classes
        features: list of str
            The feature names
        label: str
            The label name
        Returns
        -------
        Dataset
        """
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, n_classes, n_samples)
        return cls(X, y, features=features, label=label)


if __name__ == '__main__':
    import si.io.CSV as CSV

    temp = CSV.read_csv('D:/Mestrado/2ano/1semestre/SIB/si/datasets/iris/iris.csv', ',', True)

    x = np.array([[np.nan, 1, 3], [3, 2, 3], [3, np.nan, 3]])
    y = np.array([1, 2, 5])
    features = ['A', 'B', 'C']
    label = 'y'
    dataset = Dataset(X=x, y=y, features=features, label=None)
    dataset.fill_Na(0)
    print(dataset)
    # dataset= temp
    # print(dataset.get_var())

