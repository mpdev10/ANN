import numpy as np

from activation.Activation import Activation


class Softmax(Activation):
    def compute(self, z):
        epsilon = 1e-5
        max_val = np.max(z)
        numerator = np.exp(z - max_val)
        denominator = np.sum(numerator, axis=1, keepdims=True)
        return numerator / (denominator + epsilon)

    def compute_deriv(self, a):
        return 1

    def get_name(self):
        return 'softmax-stable'
