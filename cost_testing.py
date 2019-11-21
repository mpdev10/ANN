from Activation.sigmoid import Sigmoid
from Activation.softmax import Softmax
from Loss.crossentropy import CrossEntropy
from Loss.mse import MeanSquaredError
from Tests.cost_tests import perform_cost_and_last_layer_tests

cost_experiments = [
    [CrossEntropy(), Softmax()],
    [CrossEntropy(), Sigmoid()],
    [MeanSquaredError(), Softmax()],
    [MeanSquaredError(), Sigmoid()]
]

perform_cost_and_last_layer_tests(cost_experiments)
