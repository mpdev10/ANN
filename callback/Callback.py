class Callback(object):
    def on_train_start(self, model):
        pass

    def on_train_end(self, model):
        pass

    def on_batch_start(self, model):
        pass

    def on_batch_end(self, model):
        pass

    def on_epoch_start(self, model):
        pass

    def on_epoch_end(self, model):
        pass

    def on_validation_start(self, model):
        pass

    def on_validation_end(self, model):
        pass

    def on_test_start(self, model):
        pass

    def on_test_end(self, model):
        pass
