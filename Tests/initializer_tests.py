from Activation.relu import ReLU
from Activation.softmax import Softmax
from Callback.logger_callback import LoggerCallback
from Callback.plot_callback import PlotCallback
from Layer.dense import Dense
from Loss.crossentropy import CrossEntropy
from Model.model import ANN
from Optimizer.gradient_descent_static import GradientDescent
from Tests.default_config import default_parameters, X_train, y_train, X_val, y_val, X_test, y_test


def test_single_initializer(initializer):
    model = ANN(
        optimizer=GradientDescent(),
        loss=CrossEntropy(),
        layers=[
            Dense(layer_size=50, activation_func=ReLU(), weight_initializer=initializer),
            Dense(layer_size=10, activation_func=Softmax(), weight_initializer=initializer)
        ],
        callbacks=[
            LoggerCallback(),
            PlotCallback(f'./lab_3/initializers/{initializer.get_name()}')
        ]
    )

    model.fit(x_train=X_train, y_train=y_train, x_val=X_val, y_val=y_val,
              epochs=default_parameters['epochs'], batch_size=default_parameters['batch_size'])

    model.test(X_test, y_test)


def perform_initializer_test(initializer_list):
    for initializer in initializer_list:
        test_single_initializer(initializer)
