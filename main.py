import pickle as pkl
import numpy as np

from ANN import ANN


def load_file(filename):
    file = open(filename, 'rb')
    return pkl.load(file, encoding='latin1')


def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))


def init_network_from_file(loaded_file, hidden_layer_size=200):
    training_data, validation_data, test_data = loaded_file
    training_x, training_y = training_data
    label_num = np.max(training_y) + 1
    input_size = np.shape(training_x)[1]
    print("Initialized network:")
    print("Input size:", input_size, "Label number:", label_num)
    return ANN([input_size, hidden_layer_size, label_num], activation=sigmoid, activation_deriv=sigmoid_prime, mu=0,
               sigma=0.1)


if __name__ == '__main__':
    file = load_file("mnist.pkl")

    train_data, test_data, _ = file

    train_x = train_data[0]
    train_y = train_data[1]

    network = init_network_from_file(file, hidden_layer_size=30)
    network.train(train_data, 10000, 100, 0.1, test_data=test_data)
