from Layer.layer import Layer


class Softmax(Layer):
    def __init__(self, layer_size, weight_initializer, activation_func):
        super().__init__(layer_size, weight_initializer, activation_func)

    def feed(self, x):
        raise NotImplementedError

    def update(self, a):
        raise NotImplementedError
