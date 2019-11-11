import pickle as pkl

from ActivationImpl import ActivationImpl
from TestRun import TestRun

EPOCH_NUM = 40
LEARNING_RATE = 0.01
SIGMA = 1
MU = 0
RUN_NUMBER = 5
HIDDEN_LAYER_SIZE = 100
BATCH_SIZE = 40


def load_file(filename):
    file = open(filename, 'rb')
    return pkl.load(file, encoding='latin1')


def test_batch_size(testrun, batch_sizes):
    results = []
    for batch_size in batch_sizes:
        results.append(testrun.run(epoch_num=EPOCH_NUM, batch_size=batch_size, activation=ActivationImpl.relu,
                                   learning_rate=LEARNING_RATE, hidden_layer_size=HIDDEN_LAYER_SIZE, sigma=SIGMA,
                                   mu=MU))
    return results


def test_weights_range(testrun, sigmas):
    results = []
    for sigma in sigmas:
        results.append(testrun.run(epoch_num=EPOCH_NUM, batch_size=BATCH_SIZE, activation=ActivationImpl.relu,
                                   learning_rate=LEARNING_RATE, hidden_layer_size=HIDDEN_LAYER_SIZE, sigma=sigma,
                                   mu=MU))
    return results


def test_hidden_layer_size(testrun, hidden_layer_sizes):
    results = []
    for layer_size in hidden_layer_sizes:
        results.append(testrun.run(epoch_num=EPOCH_NUM, batch_size=BATCH_SIZE, activation=ActivationImpl.relu,
                                   learning_rate=LEARNING_RATE, hidden_layer_size=layer_size, sigma=SIGMA,
                                   mu=MU))
    return results


def test_activations(testrun, activations):
    results = []
    for activation in activations:
        results.append(testrun.run(epoch_num=EPOCH_NUM, batch_size=BATCH_SIZE, activation=activation,
                                   learning_rate=LEARNING_RATE, hidden_layer_size=HIDDEN_LAYER_SIZE, sigma=SIGMA,
                                   mu=MU))
    return results


if __name__ == '__main__':
    file = load_file("mnist.pkl")
    activation = ActivationImpl.sigmoid
    test_run = TestRun(file)

    batch_results = test_batch_size(test_run, [2, 50, 250, 5000, 10000, 50000])
    batch_output = open('batch.pkl', 'wb')
    pkl.dump(batch_results, batch_output)
    batch_output.close()

    weight_results = test_weights_range(test_run, [0.1, 0.5, 1, 2, 5, 10])
    weight_output = open('weight.pkl', 'wb')
    pkl.dump(weight_results, weight_output)
    weight_output.close()

    layers_results = test_hidden_layer_size(test_run, [5, 25, 50, 100, 200, 300])
    layers_output = open('layers.pkl', 'wb')
    pkl.dump(layers_results, layers_output)
    layers_output.close()

    activation_results = test_activations(test_run, [ActivationImpl.relu, ActivationImpl.sigmoid])
    activation_output = open('activation.pkl', 'wb')
    pkl.dump(activation_results, activation_output)
    activation_output.close()
