import numpy as np

from si.data.dataset import Dataset
from si.metrics.mse import mse
from sklearn.preprocessing import StandardScaler


class RidgeRegression:
    """
    The RidgeRegression is a linear model using the L2 regularization.
    This model solves the linear regression problem using an adapted Gradient Descent technique

    Parameters
    ----------
    l2_penalty: float
        The L2 regularization parameter
    alpha: float
        The learning rate
    max_iter: int
        The maximum number of iterations

    Attributes
    ----------
    theta: np.array
        The model parameters, namely the coefficients of the linear model.
        For example, x0 * theta[0] + x1 * theta[1] + ...
    theta_zero: float
        The model parameter, namely the intercept of the linear model.
        For example, theta_zero * 1
    """
    def __init__(self, use_adaptive_alpha: bool = False, l2_penalty: float = 1, alpha: float = 0.001, max_iter: int = 2000):
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

    def _regular_fit(self, dataset: Dataset) -> 'RidgeRegression':
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
            y_pred = np.dot(dataset.X, self.theta) + self.theta_zero

            # computing and updating the gradient with the learning rate
            gradient = (self.alpha * (1 / m)) * np.dot(y_pred - dataset.Y, dataset.X)

            # computing the penalty
            penalization_term = self.alpha * (self.l2_penalty / m) * self.theta

            # updating the model parameters
            self.theta = self.theta - gradient - penalization_term
            self.theta_zero = self.theta_zero - (self.alpha * (1 / m)) * np.sum(y_pred - dataset.Y)

            # custo
            custo = self.cost(dataset)
            if i == 0:
                self.history[i] = custo
            else:
                if np.abs(self.history.get(i - 1) - custo) >= 1:
                    self.history[i] = custo
                else:
                    break
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
            y_pred = np.dot(dataset.X, self.theta) + self.theta_zero

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
                dif = (self.history.get(i - 1) - custo)

                if dif < 1:
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
        return np.dot(dataset.X, self.theta) + self.theta_zero

    def score(self, dataset: Dataset) -> float:
        """
        Compute the Mean Square Error of the model on the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the MSE on

        Returns
        -------
        mse: float
            The Mean Square Error of the model
        """
        y_pred = self.predict(dataset)
        return mse(dataset.Y, y_pred)

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
        y_pred = self.predict(dataset)
        return (np.sum((y_pred - dataset.Y) ** 2) + (self.l2_penalty * np.sum(self.theta ** 2))) / (2 * len(dataset.Y))


if __name__ == '__main__':
    # import dataset
    from si.data.dataset import Dataset
    from si.io.CSV import read_csv


    data1 = read_csv("D:/Mestrado/2ano/1semestre/SIB/si/datasets/cpu/cpu.csv", ",", True, True)
    # make a linear dataset
    # X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    # y = np.dot(X, np.array([1, 2])) + 3
    # dataset_ = Dataset(X=X, y=y)

    # fit the model
    model = RidgeRegression(True)
    model.fit(data1, True)


    # get coefs
    print(f"Parameters: {model.history}")

    # compute the score
    score = model.score(data1)
    print(f"Score: {score}")

    # compute the cost
    cost = model.cost(data1)
    print(f"Cost: {cost}")

    # predict
    y_pred_ = model.predict(data1)
    print(f"Predictions: {y_pred_}")
