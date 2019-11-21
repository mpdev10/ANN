import numpy as np

from Activation.activation import Activation


class Sigmoid(Activation):
    name = 'sigmoid'

    def run(self, z):
        return 1 / (1 + np.exp(-z))

    def derivative(self, a):
        return a * (1 - a)

    def get_name(self):
        return 'sigmoid'
