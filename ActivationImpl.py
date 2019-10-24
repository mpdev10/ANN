import numpy as np

from ActivationFunction import ActivationFunction


class ActivationImpl:
    sigmoid = ActivationFunction.of(lambda x: 1.0 / (1.0 + np.exp(-x)),
                                    lambda x: (1.0 / (1.0 + np.exp(-x))) * (1 - (1.0 / (1.0 + np.exp(-x)))))

    relu = ActivationFunction.of(lambda x: np.max(0, x), lambda x: np.max(0, np.min(1, x)))
