from Activation.relu import ReLU
from Callback.logger_callback import LoggerCallback
from Callback.plot_callback import PlotCallback
from Init.xavier_initializer import XavierInit
from Layer.dense import Dense
from Model.model import ANN
from Optimizer.gradient_descent_static import GradientDescent
from Tests.default_config import default_parameters, X_train, y_train, X_val, y_val, X_test, y_test


def test_single_cost_and_last_layer(cost_func, last_layer):
    model = ANN(
        optimizer=GradientDescent(),
        loss=cost_func,
        layers=[
            Dense(layer_size=50, activation_func=ReLU(), weight_initializer=XavierInit()),
            Dense(layer_size=10, activation_func=last_layer, weight_initializer=XavierInit())
        ],
        callbacks=[
            LoggerCallback(),
            PlotCallback(f'./lab_3/cost/func={cost_func.get_name()}&last_layer={last_layer.get_name()}')
        ]
    )

    model.fit(x_train=X_train, y_train=y_train, x_val=X_val, y_val=y_val,
              epochs=default_parameters['epochs'], batch_size=default_parameters['batch_size'])

    model.test(X_test, y_test)


def perform_cost_and_last_layer_tests(data_list):
    for data in data_list:
        test_single_cost_and_last_layer(data[0], data[1])
