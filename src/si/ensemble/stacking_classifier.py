from si.data.dataset import Dataset
import numpy as np
from si.metrics.accuracy import accuracy


class StackingClassifier:
    """
    Usa uma lista de instâncias do modelo de classificação para gerar previsões. Essas previsões são
    então usadas para treinar um modelo final e auxiliar o modelo na geração das previsões.
    """

    def __init__(self, models: list, final_mod):
        """
        Guarda as variáveis e inicia a classe
        :param models: Uma lista de instâncias do modelo de classificação
        :param final_mod: Um modelo de classificação para treinar e prever valores output
                            usando os outputs fornecidos pelos modelos escolhidos.
        """
        self.models = models  # lista de modelos já inicializados
        self.final_mod = final_mod

    def fit(self, dataset):
        """
        Ajusta os modelos escolhidos usando o conjunto de dados fornecidos e usa os outputs previstos
        para ajudar a ajustar o modelo final.

        :param dataset: instância da classe Dataset para treinar cada modelo
        """
        dt = Dataset(dataset.X, dataset.Y, dataset.Features, dataset.Label)
        for model in self.models:
            model.fit(dataset)
            dt.X = np.c_[dt.X, model.predict(dataset)]

        self.final_mod.fit(dt)
        return self

    def predict(self, dataset):
        """
        Gera previsões com os modelos escolhidos e usaos outputs para
        ajudar o modelo final escolhido a fazer as suas previsões. Retorna essas previsões finais

        :param dataset: instância da classe Dataset para prever a variável dependente

        """
        dt = Dataset(dataset.X, dataset.Y, dataset.Features, dataset.Label)
        for model in self.models:
            dt.X = np.c_[dt.X, model.predict(dataset)]

        return self.final_mod.predict(dt)

    def score(self, dataset):
        """
        Retorna o score de accuracy entre os valores de y verdadeiros e previstos.

        :param dataset: instância da classe Dataset para prever a variável dependente
        """
        y_pred_ = self.predict(dataset)
        return accuracy(dataset.Y, y_pred_)
