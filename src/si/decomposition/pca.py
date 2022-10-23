import numpy as np

class PCA:
    '''
    Principal Component Analysis (PCA)
    '''

    def __init__(self, n_components:int):
        self.n_components= n_components

    def fit(self, dataset):
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

    def transform(self, dataset):
        V = self.comp_princ.T # matriz transporta
        # SVD reduced
        Xreduced = np.dot(self.cent_data, V)

        return Xreduced



