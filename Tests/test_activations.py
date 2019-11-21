from Activation.softmax import Softmax
from Callback.logger_callback import LoggerCallback
from Callback.plot_callback import PlotCallback
from Init.xavier_initializer import XavierInit
from Layer.dense import Dense
from Loss.crossentropy import CrossEntropy
from Model.model import ANN
from Optimizer.gradient_descent_static import GradientDescent
from Tests.default_config import default_parameters, X_train, y_train, X_val, y_val, X_test, y_test


def test_single_activation_function(activation):
    model = ANN(
        optimizer=GradientDescent(learning_rate=default_parameters['learning_rate']),
        loss=CrossEntropy(),
        layers=[
            Dense(layer_size=50, activation_func=activation, weight_initializer=XavierInit()),
            Dense(layer_size=10, activation_func=Softmax(), weight_initializer=XavierInit())
        ],
        callbacks=[
            LoggerCallback(),
            PlotCallback(f'./results/activations/{activation.get_name()}')
        ]
    )

    model.fit(x_train=X_train, y_train=y_train, x_val=X_val, y_val=y_val,
              epochs=default_parameters['epochs'], batch_size=default_parameters['batch_size'])

    model.test(X_test, y_test)


def test_activation_functions(activation_functions):
    for activation in activation_functions:
        test_single_activation_function(activation)
