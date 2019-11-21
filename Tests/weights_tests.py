from Activation.relu import ReLU
from Activation.softmax import Softmax
from Callback.logger_callback import LoggerCallback
from Callback.plot_callback import PlotCallback
from Layer.dense import Dense
from Loss.crossentropy import CrossEntropy
from Model.model import ANN
from Optimizer.gradient_descent_static import GradientDescent
from Tests.default_config import default_parameters, X_train, y_train, X_val, y_val, X_test, y_test


def test_single_weight_initializer(weight_initializer):
    model = ANN(
        optimizer=GradientDescent(learning_rate=default_parameters['learning_rate']),
        loss=CrossEntropy(),
        layers=[
            Dense(layer_size=50, activation_func=ReLU(), weight_initializer=weight_initializer),
            Dense(layer_size=10, activation_func=Softmax(), weight_initializer=weight_initializer)
        ],
        callbacks=[
            LoggerCallback(),
            PlotCallback(f'./results/weigh_initializer/{weight_initializer.get_name()}')
        ]
    )

    model.fit(x_train=X_train, y_train=y_train, x_val=X_val, y_val=y_val,
              epochs=default_parameters['epochs'], batch_size=default_parameters['batch_size'])

    model.test(X_test, y_test)


def test_weight_initializers(initializers):
    for initializer in initializers:
        test_single_weight_initializer(initializer)
