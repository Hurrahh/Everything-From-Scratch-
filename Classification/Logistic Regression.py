import numpy as np
from sklearn.datasets import make_classification
import pandas as pd
from sklearn.metrics import accuracy_score
class LogisticRegression():
    def __init__(self,learning_rate,epochs):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weight = None
        self.bias = None

    def fit(self,X,Y):
        self.samples, self.features = X.shape

        self.weight = np.zeros(self.features)
        self.bias = 0

        for i in range(self.epochs):
            y_predicted = 1 / (1 + np.exp(- (X.dot(self.weight) + self.bias)))

            dw = 1/(self.samples)*np.dot(X.T,(y_predicted - Y))
            db = 1/(self.samples)*np.sum(y_predicted-Y)


            self.weight = self.weight - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db

    def predict(self,X):
        y_pred = 1/(1+np.exp(- (X.dot(self.weight) + self.bias)))
        return np.where(y_pred > 0.5, 1, 0)



model = LogisticRegression(0.01,100)

X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

df = pd.DataFrame(X, columns=[f'Feature_{i+1}' for i in range(X.shape[1])])
df['Target'] = y


print(df.head())

model.fit(X,y)
predictions = model.predict(X)
print(accuracy_score(y,predictions))
