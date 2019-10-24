from sklearn.utils import shuffle

import numpy as np


class ANN:

    def __init__(self, layer_sizes, sigma=1, mu=0, activation=None, activation_deriv=None):
        self.layer_num = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.weights = [np.random.normal(loc=mu, scale=sigma, size=(y, x)) for x, y in
                        zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [np.random.normal(loc=mu, scale=sigma, size=(y, 1)) for y in layer_sizes[1:]]
        self.activation = activation
        self.activation_deriv = activation_deriv
        if self.activation is None:
            self.activation = lambda x: np.maximum(0, x)  # default - relu
            self.activation_deriv = lambda x: np.maximum(0, np.minimum(1, x))

    def forward(self, x):
        x = x.T
        for w, b in zip(self.weights, self.biases):
            x = self.activation(np.dot(w, x) + b)
        return x

    def train(self, train_data, epoch_num, batch_size, learning_rate, test_data=None):
        for i in range(epoch_num):
            batches = self._get_batches(train_data, batch_size)

            for batch in batches:
                x, y = batch
                d_w, d_b = self.backprop(x, y)
                self.weights = [w + (learning_rate * gw) for w, gw in zip(self.weights, d_w)]
                self.biases = [b + (learning_rate * gb) for b, gb in zip(self.biases, d_b)]

            if test_data:
                print("Epoch {0}: {1}/{2}".format(i, self.eval(test_data), len(test_data[1])))

    def eval(self, test_data):
        x, y = test_data
        test_results = self.forward(x)
        test_results = np.argmax(test_results, axis=0)
        return np.sum((test_results == y) * 1)

    def backprop(self, x, y):

        batch_size = len(x)
        x = x.T

        d_w = [np.zeros(w.shape) for w in self.weights]
        d_b = [np.zeros(b.shape) for b in self.biases]

        activation = x
        activations = [x]
        z_vector = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            z_vector.append(z)
            activation = self.activation(z)
            activations.append(activation)

        y_one_hot = self.one_hot(y, self.layer_sizes[-1])
        delta = (y_one_hot - activations[-1]) * self.activation_deriv(z_vector[-1])
        d_w[-1] = delta.dot(activations[-2].T) / float(batch_size)
        d_b[-1] = np.mean(delta, axis=1, keepdims=True)

        for l in range(2, self.layer_num):
            z = z_vector[-l]
            delta = np.dot(self.weights[-l + 1].T, delta) * self.activation_deriv(z)
            activation = activations[-l - 1]
            d_w[-l] = np.dot(delta, activation.T) / float(batch_size)
            d_b[-l] = np.mean(delta, axis=1, keepdims=True)

        return d_w, d_b

    @staticmethod
    def one_hot(a, num_classes):
        return np.squeeze(np.eye(num_classes)[a.reshape(-1)]).T

    @staticmethod
    def _cost_deriv(output_y, real_y):
        return output_y - real_y

    @staticmethod
    def _get_batches(train_data, batch_size):
        train_x, train_y = shuffle(train_data[0], train_data[1])
        n = len(train_data[1])
        batches = [(train_x[k:k + batch_size], train_y[k:k + batch_size]) for k in range(0, n, batch_size)]
        return batches
