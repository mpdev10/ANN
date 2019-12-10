import numpy as np
from numpy.lib.stride_tricks import as_strided

from layer.Layer import Layer


class MaxPool2d(Layer):
    def __init__(self, pool_size, stride, layer_name='max-pooling-2d'):
        super().__init__(layer_name)
        self._size = pool_size
        self._stride = stride
        self._mask = np.array([])

    def __call__(self, previous_layer_shape, optimizer, calc_error=True):
        nH, nW = Layer.compute_output_shape(previous_layer_shape, self._size, self._stride)
        print(previous_layer_shape)
        num_of_channels = previous_layer_shape[2]
        self._error_shape = (-1, nH, nW, 1, 1, num_of_channels)
        self._output_shape = (nH, nW, num_of_channels)
        return self._output_shape

    @staticmethod
    def _pooling(strided_layer_view):
        layer = np.max(strided_layer_view, axis=(-3, -2), keepdims=True).astype(np.float16)
        layer_ = layer + np.random.uniform(0, 1e-8, size=strided_layer_view.shape).astype(np.float16)
        mask = (layer_ == np.max(layer_, axis=(-3, -2), keepdims=True)).astype(np.float16)
        return mask, np.squeeze(layer, axis=(-3, -2)).astype(np.float16)

    def feed(self, input_layer):
        self._input_layer = input_layer.astype(np.float16)
        strided_layer_view = self._get_strided_layer_view(input_layer)
        self._mask, self._output_layer = self._pooling(strided_layer_view)
        return self._output_layer

    def get_error(self):
        error = np.zeros(self._input_layer.shape, dtype=np.float16)
        strided_error_view = self._get_strided_layer_view(error)
        return strided_error_view.astype(np.float16)

    def update_delta(self, error):
        mask = self._mask * error.reshape(self._error_shape)
        return self.get_error() + mask

    def _get_strided_layer_view(self, input_layer):
        num_of_inputs = input_layer.shape[0]
        mH, mW, num_of_channels = self._output_shape
        pH, pW = self._size
        stride_height, stride_width = self._stride
        stride_1, stride_2, stride_3, stride_4 = input_layer.strides
        view_shape = (num_of_inputs, mH, mW, pH, pW, num_of_channels)
        strides_shape = (stride_1, stride_2 * stride_height, stride_3 * stride_width, stride_2, stride_3, stride_4)
        return as_strided(input_layer, shape=view_shape, strides=strides_shape, writeable=False).astype(
            np.float16)
