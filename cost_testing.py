from Tests.cost_tests import perform_cost_and_last_layer_tests

from Activation.Sigmoid import Sigmoid
from Activation.Softmax import Softmax
from Loss.CrossEntropy import CrossEntropy
from Loss.MeanSquaredError import MeanSquaredError

cost_experiments = [
    [CrossEntropy(), Softmax()],
    [CrossEntropy(), Sigmoid()],
    [MeanSquaredError(), Softmax()],
    [MeanSquaredError(), Sigmoid()]
]

perform_cost_and_last_layer_tests(cost_experiments)
