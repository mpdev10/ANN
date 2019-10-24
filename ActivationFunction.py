class ActivationFunction:

    def __init__(self, activation, derivative):
        self._activation = activation
        self._derivative = derivative

    @classmethod
    def of(cls, activation, derivative):
        return cls(activation, derivative)

    def compute(self, x):
        return self._activation(x)

    def compute_derivative(self, x):
        return self._derivative(x)
