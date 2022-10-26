from si.statistics.sigmoid_function import sigmoid_function
import numpy as np
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class LogisticRegression:

    def __init__(self,use_adaptive_alpha: bool = False, l2_penalty: float = 1, alpha: float = 0.001, max_iter: int = 2000):
        """

        Parameters
        ----------
        l2_penalty: float
            The L2 regularization parameter
        alpha: float
            The learning rate
        max_iter: int
            The maximum number of iterations
        """
        # parameters
        self.l2_penalty = l2_penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.use_adaptive_alpha = use_adaptive_alpha

        # attributes
        self.theta = None
        self.theta_zero = None
        self.history = {}

    def fit(self, dataset: Dataset, STscale: bool = False):

        if STscale:
            dataset.X = StandardScaler().fit_transform(dataset.X)
        if self.use_adaptive_alpha is True: self._adaptive_fit(dataset)
        elif self.use_adaptive_alpha is False: self._regular_fit(dataset)


    def _regular_fit(self, dataset: Dataset) -> 'LogisticRegression':
        """
        Fit the model to the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to

        Returns
        -------
        self: LogisticRegression
            The fitted model
        """

        m, n = dataset.shape()
        # initialize the model parameters
        self.theta = np.zeros(n)
        self.theta_zero = 0

        # gradient descent
        for i in range(self.max_iter):
            # predicted y
            y_pred = sigmoid_function(np.dot(dataset.X, self.theta) + self.theta_zero)

            # computing and updating the gradient with the learning rate
            gradient = (self.alpha * (1 / m)) * np.dot(y_pred - dataset.Y, dataset.X)

            # computing the penalty
            penalization_term = self.alpha * (self.l2_penalty / m) * self.theta

            # updating the model parameters
            self.theta = self.theta - gradient - penalization_term
            self.theta_zero = self.theta_zero - (self.alpha * (1 / m)) * np.sum(y_pred - dataset.Y)

            #custo
            custo = self.cost(dataset)
            if i == 0:
                self.history[i] = custo
            else:
                if np.abs(self.history.get(i-1) - custo) >= 0.0001:
                    self.history[i] = custo
                else:
                    break
            # self.history[i] = custo


        return self

    def _adaptive_fit(self, dataset: Dataset) -> 'LogisticRegression':
        """
        Fit the model to the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to

        Returns
        -------
        self: RidgeRegression
            The fitted model
        """

        m, n = dataset.shape()

        # initialize the model parameters
        self.theta = np.zeros(n)
        self.theta_zero = 0


        # gradient descent
        for i in range(self.max_iter):

            # predicted y
            y_pred = sigmoid_function(np.dot(dataset.X, self.theta) + self.theta_zero)

            # computing and updating the gradient with the learning rate
            gradient = (self.alpha * (1 / m)) * np.dot(y_pred - dataset.Y, dataset.X)

            # computing the penalty
            penalization_term = self.alpha * (self.l2_penalty / m) * self.theta

            # updating the model parameters
            self.theta = self.theta - gradient - penalization_term
            self.theta_zero = self.theta_zero - (self.alpha * (1 / m)) * np.sum(y_pred - dataset.Y)

            #custo
            custo = self.cost(dataset)

            if i != 0:
                dif = (self.history.get(i-1) - custo)

                if dif < 0.0001:
                    self.alpha = self.alpha / 2

            self.history[i] = custo

        no_dups = {}

        for key, value in self.history.items():
            if value not in no_dups.values():
                no_dups[key] = value

        self.history = no_dups

        return self

    def line_plot(self):

        it = list(self.history.keys())  # list() needed for python 3.x
        custo = list(self.history.values())  # ditto
        plt.plot(it, custo, '-')

        return plt.show()

    def predict(self, dataset: Dataset) -> np.array:
        """
        Predict the output of the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the output of

        Returns
        -------
        predictions: np.array
            The predictions of the dataset
        """

        values = sigmoid_function(np.dot(dataset.X, self.theta) + self.theta_zero)
        arr = []

        for i in values:
            if i >= 0.5:
                arr.append(1)
            elif i < 0.5:
                arr.append(0)
        return np.array(arr)

    def score(self, dataset: Dataset) -> float:
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
        y_pred_= self.predict(dataset)
        return accuracy(dataset.Y, y_pred_)

    def cost(self, dataset: Dataset) -> float:
        """
        Compute the cost function (J function) of the model on the dataset using L2 regularization

        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the cost function on

        Returns
        -------
        cost: float
            The cost function of the model
        """
        prediction = sigmoid_function(np.dot(dataset.X, self.theta) + self.theta_zero)
        cost = (-dataset.Y + np.log(prediction)) - ((1 - dataset.Y) + np.log(1 - prediction))
        cost = np.sum(cost) / dataset.shape()[0]
        cost = cost + (self.l2_penalty * np.sum(self.theta ** 2) / (2 * dataset.shape()[0]))
        return cost


if __name__ == '__main__':
    import dataset
    from si.model_selection.split import train_test_split
    from si.io.CSV import read_csv
    from sklearn.preprocessing import StandardScaler
    data1 = read_csv("D:/Mestrado/2ano/1semestre/SIB/si/datasets/breast/breast-bin.data", ",", False, True)




    # fit the model
    model = LogisticRegression(True)
    model.fit(data1, True)
    model.line_plot()
    # print(model.history)


    # # get coefs
    # print(f"Parameters: {model.theta}")
    #
    # # compute the score
    # score = model.score(data1)
    # print(f"Score: {score}")

    # compute the cost
    # cost = model.cost(dataset_)
    # print(f"Cost: {cost}")

    # predict
    # y_pred_ = model.predict(data1)
    # print()
    # print(f"Predictions: {y_pred_}")
    # print('\nScore:',model.score(data1) )
    # print('\nCost:', model.cost(data1))

