from Callback.logger_callback import LoggerCallback
from Callback.plot_callback import PlotCallback
from Loss.crossentropy import CrossEntropy
from Model.model import ANN
from Optimizer.gradient_descent_static import GradientDescent
from Tests.default_config import default_parameters, X_train, y_train, X_val, y_val, X_test, y_test


def test_single_layer(layer_config):
    model = ANN(
        optimizer=GradientDescent(learning_rate=default_parameters['learning_rate']),
        loss=CrossEntropy(),
        layers=layer_config['Layer'],
        callbacks=[
            LoggerCallback(),
            PlotCallback(f"./results/Layer/{layer_config['name']}")
        ]
    )

    model.fit(x_train=X_train, y_train=y_train, x_val=X_val, y_val=y_val,
              epochs=default_parameters['epochs'], batch_size=default_parameters['batch_size'])

    model.test(X_test, y_test)


def test_layer_configs(layer_configs):
    for layer_config in layer_configs:
        test_single_layer(layer_config)
