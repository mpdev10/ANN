import numpy as np

from Init.Init import Init


class ZerosInit(Init):
    def __call__(self, shape):
        return np.zeros(shape=shape)

    def get_name(self):
        return 'zeros-initializer'
