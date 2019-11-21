from Activation.relu import ReLU
from Activation.softmax import Softmax
from Callback.logger_callback import LoggerCallback
from Callback.plot_callback import PlotCallback
from Init.xavier_initializer import XavierInit
from Layer.dense import Dense
from Loss.crossentropy import CrossEntropy
from Model.model import ANN
from Tests.default_config import default_parameters, X_train, y_train, X_val, y_val, X_test, y_test


def test_single_optimizer(optimizer):
    model = ANN(
        optimizer=optimizer,
        loss=CrossEntropy(),
        layers=[
            Dense(layer_size=50, activation_func=ReLU(), weight_initializer=XavierInit()),
            Dense(layer_size=10, activation_func=Softmax(), weight_initializer=XavierInit())
        ],
        callbacks=[
            LoggerCallback(),
            PlotCallback(f'./lab_3/optimizers/{optimizer.get_name()}')
        ]
    )

    model.fit(x_train=X_train, y_train=y_train, x_val=X_val, y_val=y_val,
              epochs=default_parameters['epochs'], batch_size=default_parameters['batch_size'])

    model.test(X_test, y_test)


def perform_optimizer_test(optimizer_list):
    for optimizer in optimizer_list:
        test_single_optimizer(optimizer)
