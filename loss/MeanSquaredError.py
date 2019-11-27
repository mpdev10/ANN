import numpy as np

from loss.Loss import Loss


class MeanSquaredError(Loss):
    def __call__(self, y, y_pred):
        error = y - y_pred
        cost = np.mean(np.square(error))
        error /= y.shape[0]
        return error, cost

    def get_name(self):
        return 'MSE'
