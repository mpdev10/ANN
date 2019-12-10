class Layer(object):
    def __init__(self, layer_name='abstract-layer'):
        self._name = layer_name

    def __call__(self, previous_layer_size, optimizer, calc_error=True):
        raise NotImplementedError

    def update_delta(self, error):
        raise NotImplementedError

    def feed(self, x):
        raise NotImplementedError

    def get_name(self):
        return self._name

    @staticmethod
    def compute_output_shape(input_layer, size, stride):
        result = []
        for i in range(len(input_layer) - 1):
            result.append(int((input_layer[i] - size[i]) / stride[i]) + 1)
        return result
