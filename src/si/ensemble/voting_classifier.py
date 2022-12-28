import numpy as np
from si.metrics.accuracy import accuracy
from si.data.dataset import Dataset
class VotingClassifier:
    """
    Testa vários modelos dados como input classificando-os com base na maioria de votos de modo a prever os labels.
    """

    def __init__(self, models: list):
        """
        Guarda as variáveis

        :param models: Uma lista de algoritmos de machine learning para testar em dados fornecidos
        """
        self.models = models #lista de modelos já inicializados

    def fit(self, dataset):
        """
        Treina o modelo comos dados fornecidos.

        :param dataset: Intsância da classe Dataset que servirá para treinar cada modelo
        """
        for model in self.models:
            model.fit(dataset)

        return self

    def predict(self, dataset):
        """
        Usa cada modelo para prever a variável dependente, retornando um array com os outputs mais votados

        :param dataset: Intsância da classe Dataset que servirá para prever a variável dependente.
        """

        def _get_majority_vote(pred:np.array) -> int:
            # Fornecido pelo Prof
            """
            It returns the majority vote of the given predictions
            Parameters
            ----------
            pred: np.ndarray
                The predictions to get the majority vote of
            Returns
            -------
            majority_vote: int
                The majority vote of the given predictions
            """
            labels, counts = np.unique(pred, return_counts = True)
            return labels[np.argmax(counts)]

        vote = np.array([model.predict(dataset) for model in self.models]).transpose()
        return np.apply_along_axis(_get_majority_vote, axis = 1, arr = vote)

    def score(self, dataset: Dataset) -> float:
        # Dado pelo Prof
        """
        Compute the Mean Square Error of the model on the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the MSE on

        Returns
        -------
        accuracy: float
            The Mean Square Error of the model
        """

        y_pred_ = self.predict(dataset)
        return accuracy(dataset.Y, y_pred_)

