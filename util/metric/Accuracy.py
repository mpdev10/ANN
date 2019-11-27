import numpy as np

from util.metric.Metric import Metric


class Accuracy(Metric):
    name = 'accuracy'

    @staticmethod
    def calculate(y_1, y_2):
        y_1, y_2 = np.asarray(y_1), np.asarray(y_2)
        results = np.where(np.argmax(y_1, axis=1) == np.argmax(y_2, axis=1), 1, 0)
        return np.mean(results) * 100
