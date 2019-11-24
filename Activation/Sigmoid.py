import numpy as np

from Activation.Activation import Activation


class Sigmoid(Activation):
    name = 'sigmoid'

    def compute(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_deriv(self, a):
        return a * (1 - a)

    def get_name(self):
        return 'sigmoid'
