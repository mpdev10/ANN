import copy

from ANN import ANN


class TestRun:
    def __init__(self, test_network: ANN, train_data, validation_data, test_data):
        self.test_network = test_network
        self.networks = dict()
        self.train_data = train_data
        self.validation_data = validation_data
        self.test_data = test_data

    def run(self, epoch_num, batch_range=[20, 100], batch_step=20, learning_rate=0.01):

        for batch_size in range(batch_range[0], batch_range[1], batch_step):
            self.networks[batch_size] = copy.deepcopy(self.test_network)

        for e in range(0, epoch_num):
            print("EPOCH " + e.__str__())
            for batch_size in range(batch_range[0], batch_range[1], batch_step):
                network = self.networks[batch_size]

                network.train(self.train_data, 1, batch_size, learning_rate, test_data=self.test_data)
                print(
                    "Batch size {0}: {1}/{2}".format(batch_size, network.eval(self.test_data),
                                                     len(self.validation_data[1])))
