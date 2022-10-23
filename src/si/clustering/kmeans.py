import numpy as np
from si.statistics.euclidean_distance import euclidean_distance
from si.data.dataset import Dataset

class KMeans:
    '''
    Class: clustering de dados usando o método Kmeans
    '''

    def __init__(self, k, max_iter = 300, distance = euclidean_distance):

        self.k= k
        self.max_iter = max_iter
        self.distance = distance

    def _closest_centroid(self, row, centroids):

        distances= self.distance(row, centroids)
        best_index= np.argmin(distances,axis=0)
        return best_index

    def fit(self, dataset:Dataset):

        seeds = (np.random.permutation(dataset.X.shape[0]))[:self.k]
        self.centroids = dataset.X[seeds,:]

        convergence = True
        labels = []
        cont = 0

        while convergence and cont < self.max_iter:
            new_labels = np.apply_along_axis(self._closest_centroid, axis=1, arr=dataset.X, centroids=self.centroids) #Linhas
            all_centroids = []
            for ind in range(self.k):
                centroid = dataset.X[new_labels == ind]
                cent_mean = np.mean(centroid, axis=0) #Colunas
                all_centroids.append(cent_mean)

            self.centroids = np.array(all_centroids)

            convergence = np.any(new_labels != labels)
            labels = new_labels
            cont +=1

        print(f'Iteração:{cont}')
        return self

    def transform(self, dataset:Dataset):
        return np.apply_along_axis(self.distance, axis=1, arr=dataset.X, y=self.centroids)

    def predict(self, dataset:Dataset):
        distances=self.transform(dataset)
        labels= np.argmin(distances, axis=1)
        return labels

