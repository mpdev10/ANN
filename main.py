from activation.Relu import Relu
from activation.Softmax import Softmax
from callback.LoggerCallback import LoggerCallback
from init.XavierInit import XavierInit
from layer.Conv2d import Conv2d
from layer.Dense import Dense
from layer.Flatten import Flatten
from loss.CrossEntropy import CrossEntropy
from model.MultiLayerNetwork import MultiLayerNetwork
from optimizer.Adam import Adam
from util.preprocessing.data_loader import get_data

configuration = {
    'epochs': 20,
    'batch_size': 20,
    'learning_rate': 0.0001
}

(X_train, y_train), (X_val, y_val), (X_test, y_test) = get_data()


def test_kernel_sizes(kernel_sizes):
    for kernel_s in kernel_sizes:
        model = MultiLayerNetwork(
            optimizer=Adam(learning_rate=configuration['learning_rate']),
            loss=CrossEntropy(),
            layers=[
                Conv2d(filter_num=1, kernel=kernel_s, activation_func=Relu()),
                Flatten(),
                Dense(layer_size=50, activation_func=Relu(), weight_initializer=XavierInit(gain=12)),
                Dense(layer_size=10, activation_func=Softmax(), weight_initializer=XavierInit(gain=12))
            ],
            callbacks=[LoggerCallback()]
        )
        model.fit(x_train=X_train, y_train=y_train, x_val=X_val, y_val=y_val,
                  epochs=configuration['epochs'], batch_size=configuration['batch_size'])

        model.test(X_test, y_test)


if __name__ == '__main__':
    test_kernel_sizes([[3, 3], [5, 5]])
