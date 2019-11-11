from collections import defaultdict

import numpy as np

from ANN import ANN
from TestRunResults import TestRunResults


class TestRun:
    def __init__(self, data_file):
        self._train_data, self._validation_data, self._test_data = data_file
        self._data_file = data_file

    def run(self, epoch_num, batch_size, activation, learning_rate=0.01, sigma=1, mu=0, run_num=5,
            hidden_layer_size=100):
        results = defaultdict(list)
        test_accuracy = []

        for i in range(0, run_num):
            network = self._init_network_from_file(self._data_file, activation, hidden_layer_size, sigma, mu)
            run_results = network.train(self._train_data, epoch_num, batch_size, learning_rate,
                                        test_data=self._validation_data)
            for i in range(0, len(run_results)):
                results[i].append(run_results[i])
                test_accuracy.append(network.eval(self._test_data) / len(self._test_data[1]))

        results_avg = [np.mean(results[i]) for i in range(0, epoch_num)]
        results_std = [np.std(results[i]) for i in range(0, epoch_num)]
        test_avg = np.mean(test_accuracy)
        test_std = np.std(test_accuracy)

        return TestRunResults(results_avg, results_std, test_avg, test_std)

    @staticmethod
    def _init_network_from_file(loaded_file, activation, hidden_layer_size, sigma, mu):
        training_data, validation_data, test_data = loaded_file
        training_x, training_y = training_data
        label_num = np.max(training_y) + 1
        input_size = np.shape(training_x)[1]
        return ANN([input_size, hidden_layer_size, label_num], activation=activation,
                   mu=mu,
                   sigma=sigma)
