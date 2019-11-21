import numpy as np

from Activation.activation import Activation


class ReLU(Activation):
    name = 'relu'

    def run(self, z):
        return np.where(z > 0, z, 0)

    def derivative(self, a):
        return np.where(a > 0, 1, 0)

    def get_name(self):
        return 'relu'
