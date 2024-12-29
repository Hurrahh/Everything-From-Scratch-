import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            y_predicted = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


class PolynomailRegression():

    def __init__(self, degree, learning_rate, iterations):
        self.degree = degree
        self.learning_rate = learning_rate
        self.iterations = iterations

    def transform(self, X):
        X_transform = np.ones((self.m, 1))
        j = 0
        for j in range(self.degree + 1):
            if j != 0:
                x_pow = np.power(X, j)
                X_transform = np.append(X_transform, x_pow.reshape(-1, 1), axis=1)
        return X_transform
    def normalize(self, X):
        X[:, 1:] = (X[:, 1:] - np.mean(X[:, 1:], axis=0)) / np.std(X[:, 1:], axis=0)
        return X

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.m, self.n = self.X.shape
        self.W = np.zeros(self.degree + 1)
        X_transform = self.transform(self.X)
        X_normalize = self.normalize(X_transform)
        for i in range(self.iterations):
            h = self.predict(self.X)
            error = h - self.Y
            self.W = self.W - self.learning_rate * (1 / self.m) * np.dot(X_normalize.T, error)
        return self

    def predict(self, X):
        X_transform = self.transform(X)
        X_normalize = self.normalize(X_transform)
        return np.dot(X_transform, self.W)




np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(0, 1)

y = y.reshape(-1)
model = LinearRegression(learning_rate=0.1, epochs=1000)
model.fit(X, y)
predictions = model.predict(X)

plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, predictions, color='red', label='Regression line')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
