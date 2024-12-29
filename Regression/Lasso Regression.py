import numpy as np

class Lasso_Regression():

    def __init__(self, learning_rate, no_of_iterations, lambda_parameter):

        self.learning_rate = learning_rate
        self.features_of_iterations = no_of_iterations
        self.lambda_parameter = lambda_parameter

    def fit(self, X, Y):

        self.samples, self.features = X.shape

        self.weight = np.zeros(self.features)

        self.bias = 0

        self.X = X

        self.Y = Y


        for i in range(self.features_of_iterations):
            Y_prediction = self.predict(self.X)

            dw = np.zeros(self.features)

            for i in range(self.features):

                if self.weight[i] > 0:

                    dw[i] = (-(2 * (self.X[:, i]).dot(self.Y - Y_prediction)) + self.lambda_parameter) / self.samples

                else:

                    dw[i] = (-(2 * (self.X[:, i]).dot(self.Y - Y_prediction)) - self.lambda_parameter) / self.samples

            db = - 2 * np.sum(self.Y - Y_prediction) / self.samples


            self.weight = self.weight - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db

    def predict(self, X):

        return X.dot(self.weight) + self.bias