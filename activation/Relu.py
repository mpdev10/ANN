import numpy as np

from activation.Activation import Activation


class Relu(Activation):
    name = 'relu'

    def compute(self, z):
        return np.where(z > 0, z, 0)

    def compute_deriv(self, a):
        return np.where(a > 0, 1, 0)

    def get_name(self):
        return 'relu'
