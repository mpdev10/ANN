import time

from Callback.Callback import Callback
from Util.pickle_utils import save_file


class SaveResultsCallback(Callback):
    def __init__(self, destination_path):
        self.val_accuracies = []
        self.val_losses = []
        self.training_accuracies = []
        self.training_losses = []
        self.epochs = []
        self.path = destination_path
        self._start_time = 0

    def on_training_begin(self, model):
        self.val_accuracies = []
        self.val_losses = []
        self.training_accuracies = []
        self.training_losses = []
        self.epochs = []
        self._start_time = time.time()

    def on_epoch_end(self, model):
        self.epochs.append(model.get_training_state().current_epoch)
        self.training_accuracies.append(model.get_training_state().current_training_accuracy)
        self.training_losses.append(model.get_training_state().current_training_cost)

    def on_validation_test_end(self, model):
        self.val_accuracies.append(model.get_training_state().current_validation_accuracy)
        self.val_losses.append(model.get_training_state().current_validation_cost)

    def on_training_end(self, model):
        batch_size = model.get_training_state().batch_size
        epochs = model.get_training_state().epochs
        learning_rate = model.get_training_state().learning_rate
        time_diff = (time.time() - self._start_time)

        save_file(self.path + '/training/', f'training_acc_batch={batch_size}_epochs={epochs}_lr={learning_rate}.pkl',
                  (zip(self.epochs, self.training_accuracies), time_diff))

        save_file(self.path + '/training/',
                  f'training_losses_batch={batch_size}_epochs={epochs}_lr={learning_rate}.pkl',
                  (zip(self.epochs, self.training_losses), time_diff))

        save_file(self.path + '/validation/',
                  f'validation_costs_batch={batch_size}_epochs={epochs}_lr={learning_rate}.pkl',
                  (zip(self.epochs, self.val_losses), time_diff))

        save_file(self.path + '/validation/',
                  f'validation_accr_batch={batch_size}_epochs={epochs}_lr={learning_rate}.pkl',
                  (zip(self.epochs, self.val_accuracies), time_diff))
