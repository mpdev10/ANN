import numpy as np

from init.Init import Init


class XavierInit(Init):
    def __init__(self, gain=6):
        self._gain = gain

    def __call__(self, shape):
        bound = np.sqrt(self._gain / (shape[0] + shape[1]))
        return np.random.uniform(-bound, bound, size=shape)

    def get_name(self):
        return f'xavier-gain={self._gain}'
