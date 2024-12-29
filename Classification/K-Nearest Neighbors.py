import numpy as np
import statistics


class KNN_Classifier():

    def __init__(self, distance_metric):
        self.distance_metric = distance_metric

    def get_distance_metric(self, training_data_point, test_data_point):

        if (self.distance_metric == 'euclidean'):
            euclidean_dist = 0
            for i in range(len(training_data_point) - 1):
                euclidean_dist = euclidean_dist + (training_data_point[i] - test_data_point[i]) ** 2

            return np.sqrt(euclidean_dist)

        elif (self.distance_metric == 'manhattan'):

            manhattan_dist = 0
            for i in range(len(training_data_point) - 1):
                manhattan_dist = manhattan_dist + abs(training_data_point[i] - test_data_point[i])

            return manhattan_dist

    def nearest_neighbors(self, X_train, test_data, k):

        distance_list = []
        for training_data in X_train:
            distance = self.get_distance_metric(training_data, test_data)
            distance_list.append((training_data, distance))

        distance_list.sort(key=lambda x: x[1])
        neighbors_list = []

        for j in range(k):
            neighbors_list.append(distance_list[j][0])

        return neighbors_list

    def predict(self, X_train, test_data, k):
        neighbors = self.nearest_neighbors(X_train, test_data, k)

        for data in neighbors:
            label = []
            label.append(data[-1])

        predicted_class = statistics.mode(label)

        return predicted_class


X = []
X_test = []
classifier = KNN_Classifier(distance_metric='euclidean')
classifier.predict(X,X_test,k=5)
