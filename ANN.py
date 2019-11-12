import numpy as np

from MathUtils import MathUtils


class ANN:

    def __init__(self, layer_sizes, activations, sigma=1, mu=0, transpose_inputs=True):
        self._layer_num = len(layer_sizes)
        self._layer_sizes = layer_sizes
        self._weights = [np.random.normal(loc=mu, scale=sigma, size=(y, x)) for x, y in
                         zip(layer_sizes[:-1], layer_sizes[1:])]
        self._biases = [np.random.normal(loc=mu, scale=sigma, size=(y, 1)) for y in layer_sizes[1:]]
        self._activations = activations
        self._transpose_inputs = transpose_inputs

    def forward(self, x):
        x = self._get_fixed_input(x)
        for idx, (w, b) in enumerate(zip(self._weights, self._biases)):
            x = self._activations[idx].compute(w.dot(x) + b)
        return x

    def train(self, train_data, epoch_num, batch_size, learning_rate, test_data=None, log=False):
        results = []
        for i in range(epoch_num):
            batches = MathUtils.partition_data(train_data, batch_size)

            for batch in batches:
                x, y = batch
                d_w, d_b = self._backprop(x, y, batch_size)
                self._weights = [w + (learning_rate * gw) for w, gw in zip(self._weights, d_w)]
                self._biases = [b + (learning_rate * gb) for b, gb in zip(self._biases, d_b)]

            accuracy = self.eval(test_data) / len(test_data[1])
            results.append(accuracy)
            if test_data and log:
                print("Epoch {0}: {1}/{2}".format(i, self.eval(test_data), len(test_data[1])))
        return results

    def eval(self, test_data):
        x, y = test_data
        test_results = self.forward(x)
        test_results = np.argmax(test_results, axis=0)
        return np.sum((test_results == y) * 1)

    def _backprop(self, x, y, batch_size):
        x = self._get_fixed_input(x)

        d_w = [np.zeros(w.shape) for w in self._weights]
        d_b = [np.zeros(b.shape) for b in self._biases]

        activation = x
        activations = [x]
        z_vector = []
        for idx, (w, b) in enumerate(zip(self._weights, self._biases)):
            z = (w.dot(activation)) + b
            activation = self._activations[idx].compute(z)
            z_vector.append(z)
            activations.append(activation)

        y_one_hot = MathUtils.one_hot(y, self._layer_sizes[-1])
        error_deriv = (y_one_hot - activations[-1])
        delta = error_deriv * self._activations[-1].compute_derivative(z_vector[-1])
        d_w[-1] = delta.dot(activations[-2].T)
        d_b[-1] = np.sum(delta, axis=1, keepdims=True)

        for l in range(2, self._layer_num):
            z = z_vector[-l]
            delta = np.dot(self._weights[-l + 1].T, delta) * self._activations[-l].compute_derivative(z)
            activation = activations[-l - 1]
            d_w[-l] = delta.dot(activation.T)
            d_b[-l] = np.sum(delta, axis=1, keepdims=True)

        return d_w, d_b

    def _get_fixed_input(self, x):
        return x.T if self._transpose_inputs else x
