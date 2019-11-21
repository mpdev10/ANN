import numpy as np

from Init.init import Init


class ZeroInit(Init):
    def __call__(self, shape):
        return np.zeros(shape=shape)

    def get_name(self):
        return 'zero-initializer'
