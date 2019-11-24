import numpy as np

from Activation.Activation import Activation


class Softmax(Activation):
    def compute(self, z):
        numerator = np.exp(z - np.max(z))
        denominator = np.sum(numerator, axis=1, keepdims=True)
        return numerator / denominator

    def compute_deriv(self, a):
        return 1

    def get_name(self):
        return 'softmax-stable'
