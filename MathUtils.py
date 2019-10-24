import numpy as np
from sklearn.utils import shuffle


class MathUtils:

    @staticmethod
    def one_hot(a, num_classes):
        return np.squeeze(np.eye(num_classes)[a.reshape(-1)]).T

    @staticmethod
    def mse_deriv(output_y, real_y):
        return output_y - real_y

    @staticmethod
    def partition_data(train_data, batch_size):
        train_x, train_y = shuffle(train_data[0], train_data[1])
        n = len(train_data[1])
        batches = [(train_x[k:k + batch_size], train_y[k:k + batch_size]) for k in range(0, n, batch_size)]
        return batches


