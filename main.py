from Activation.relu import ReLU
from Activation.softmax import Softmax
from Callback.logger_callback import LoggerCallback
from Callback.plot_callback import PlotCallback
from Callback.save_best_callback import SaveBestCallback
from Init.xavier_initializer import XavierInit
from Layer.dense import Dense
from Loss.mse import MeanSquaredError
from Model.model import ANN
from Optimizer.gradient_descent_static import GradientDescent
from Util.Preprocessing.data_loader import get_data

model = ANN(
    optimizer=GradientDescent(learning_rate=0.01),
    loss=MeanSquaredError(),
    layers=[
        Dense(layer_size=50, activation_func=ReLU(), weight_initializer=XavierInit()),
        Dense(layer_size=10, activation_func=Softmax(), weight_initializer=XavierInit())
    ],
    callbacks=[
        SaveBestCallback('./results/01_10_2019_13:00', 'best_model.pkl'),
        LoggerCallback(),
        PlotCallback('./results/batch_size/', 'test.pkl')
    ]
)

(X_train, y_train), (X_val, y_val), (X_test, y_test) = get_data()

model.fit(
    x_train=X_train, y_train=y_train,
    x_val=X_val, y_val=y_val,
    epochs=10,
    batch_size=32
)

model.test(X_test, y_test)
