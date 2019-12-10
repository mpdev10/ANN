import numpy as np

from activation.Sigmoid import Sigmoid
from init.XavierInit import XavierInit
from init.ZerosInit import ZerosInit
from layer.Layer import Layer


class Dense(Layer):

    def __init__(self, layer_size, weight_initializer=XavierInit(), activation_func=Sigmoid(),
                 bias_initializer=ZerosInit(), layer_name='dense'):
        super().__init__(layer_name)
        self._layer_size = layer_size
        self._weight_initializer = weight_initializer
        self._activation_func = activation_func
        self._bias_initializer = bias_initializer
        self._weights = None
        self._biases = None
        self._activations = None
        self._z = None
        self._delta = None
        self._optimizer = None

    def __call__(self, previous_layer_shape, optimizer, calc_error=True):
        self._weights = self._weight_initializer((previous_layer_shape, self._layer_size))
        self._biases = self._bias_initializer((1, self._layer_size))
        self._optimizer = optimizer
        self._calc_error = calc_error
        return self._layer_size

    def feed(self, x):
        self._input_layer = x
        self._z = x @ self._weights + self._biases
        self._output_layer = self._activation_func.compute(self._z)
        return self._output_layer

    def _get_delta(self, error):
        return error * self._activation_func.compute_deriv(self._z)

    def update_delta(self, error):
        delta_error = self._get_delta(error)
        if self._calc_error:
            error = self._get_error(delta_error)
        return error

    def _get_error(self, delta):
        return delta @ self._weights.T

    def update(self, x, error, cost):
        self._weights += self._optimizer.calc_gradients(id(self._weights), x.T @ self._delta)
        self._biases += self._optimizer.calc_gradients(id(self._biases),
                                                       np.sum(self._delta, axis=0, keepdims=True))
        return self._activations
