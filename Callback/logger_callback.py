from Callback.callback import Callback


class LoggerCallback(Callback):
    def __init__(self):
        self._best_validation_accuracy = 0
        self._best_validation_cost = 0

    def on_training_end(self, model):
        print(f'Finished training... Best validation result: '
              f'[Accuracy: {self._best_validation_accuracy:.2f}%] '
              f'[Cost: {self._best_validation_cost:.8f}]')

    def on_training_begin(self, model):
        print(f'Training started... ')
        self._best_validation_accuracy = 0
        self._best_validation_cost = 0
        print(f'Chosen optimizer: {model.get_optimizer().get_name()}({model.get_optimizer().get_parameters()})')

    def on_validation_test_end(self, model):
        print(f'[Epoch: {model.get_state().current_epoch:5d}, '
              f'Train accr: {model.get_state().current_training_accuracy:.2f}%, '
              f'Train cost: {model.get_state().current_training_cost:.8f}, '
              f'Val accr: {model.get_state().current_validation_accuracy:.2f}%, '
              f'Val cost: {model.get_state().current_validation_cost:.8f}]')

        if model.get_state().current_validation_accuracy > self._best_validation_accuracy:
            self._best_validation_accuracy = model.get_state().current_validation_accuracy
            self._best_validation_cost = model.get_state().current_validation_cost

    def on_test_end(self, model):
        print(f'Test accuracy: {model.get_state().test_accuracy:.2f}%'
              f' | Epochs: {model.get_state().epochs}, Batch size: {model.get_state().batch_size}')
