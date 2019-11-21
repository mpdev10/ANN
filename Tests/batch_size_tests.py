from Activation.relu import ReLU
from Activation.softmax import Softmax
from Callback.logger_callback import LoggerCallback
from Init.xavier_initializer import XavierInit
from Layer.dense import Dense
from Loss.crossentropy import CrossEntropy
from Model.model import ANN
from Optimizer.gradient_descent_static import GradientDescent
from Tests.default_config import default_parameters, X_train, y_train, X_val, y_val, X_test, y_test


def single_batch_size_test(batch_size):
    model = ANN(
        optimizer=GradientDescent(learning_rate=default_parameters['learning_rate']),
        loss=CrossEntropy(),
        layers=[
            Dense(layer_size=50, activation_func=ReLU(), weight_initializer=XavierInit()),
            Dense(layer_size=10, activation_func=Softmax(), weight_initializer=XavierInit())
        ],
        callbacks=[
            LoggerCallback(),
        ]
    )

    model.fit(x_train=X_train, y_train=y_train, x_val=X_val, y_val=y_val,
              epochs=default_parameters['epochs'], batch_size=batch_size)

    model.test(X_test, y_test)


def test_batch_sizes(batch_size_array):
    for batch_size in batch_size_array:
        single_batch_size_test(batch_size)
