from Activation.Relu import Relu
from Activation.Softmax import Softmax
from Callback.LoggerCallback import LoggerCallback
from Callback.SaveResultsCallback import SaveResultsCallback
from Init.XavierInit import XavierInit
from Layer.Dense import Dense
from Loss.CrossEntropy import CrossEntropy
from Model.MultiLayerNetwork import MultiLayerNetwork
from Optimizer.GradientDescent import GradientDescent
from Optimizer.GradientMomentum import GradientMomentum
from Tests.default_config import default_parameters, X_train, y_train, X_val, y_val, X_test, y_test


def test_single_initializer(initializer):
    model = MultiLayerNetwork(
        optimizer=GradientDescent(),
        loss=CrossEntropy(),
        layers=[
            Dense(layer_size=50, activation_func=Relu(), weight_initializer=initializer),
            Dense(layer_size=10, activation_func=Softmax(), weight_initializer=initializer)
        ],
        callbacks=[
            LoggerCallback(),
            SaveResultsCallback(f'./tests/initializers/{initializer.get_name()}')
        ]
    )

    model.fit(x_train=X_train, y_train=y_train, x_val=X_val, y_val=y_val,
              epochs=default_parameters['epochs'], batch_size=default_parameters['batch_size'])

    model.test(X_test, y_test)


def perform_initializer_test(initializer_list):
    for initializer in initializer_list:
        test_single_initializer(initializer)


def test_momentum_rates(rates):
    for rate in rates:
        model = MultiLayerNetwork(
            optimizer=GradientMomentum(learning_rate=default_parameters['learning_rate'], momentum=rate),
            loss=CrossEntropy(),
            layers=[
                Dense(layer_size=50, activation_func=Relu(), weight_initializer=XavierInit),
                Dense(layer_size=10, activation_func=Softmax(), weight_initializer=XavierInit)
            ],
            callbacks=[
                SaveResultsCallback(f'./tests/optim/momentum/rate{rate}')
            ]
        )

        model.fit(x_train=X_train, y_train=y_train, x_val=X_val, y_val=y_val,
                  epochs=default_parameters['epochs'], batch_size=default_parameters['batch_size'])
        model.test(X_test, y_test)

