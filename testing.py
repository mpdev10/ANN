from Tests.test_activations import test_activation_functions

from Activation.Relu import Relu
from Activation.Sigmoid import Sigmoid

# test_weight_initializers([
#     XavierInitializer(1),
#     XavierInitializer(6),
#     RangeInitializer(-1, 1),
#     RangeInitializer(-2, 2),
#     ZeroInitializer(),
#     RangeInitializer(-0.05, 0.05)
# ])
#
test_activation_functions([Sigmoid(), Relu()])
#
# test_layer_configs([
#     {
#         'name': 'one_layer_5_neurons',
#         'Layer': [
#             Dense(layer_size=5, activation_func=ReLu(), weight_initializer=XavierInitializer()),
#             Dense(layer_size=10, activation_func=Softmax(), weight_initializer=XavierInitializer())
#         ]
#     },
#     {
#         'name': 'one_layer_1_neurons',
#         'Layer': [
#             Dense(layer_size=1, activation_func=ReLu(), weight_initializer=XavierInitializer()),
#             Dense(layer_size=10, activation_func=Softmax(), weight_initializer=XavierInitializer())
#         ]
#     },
#     {
#         'name': 'one_layer_50_neurons',
#         'Layer': [
#             Dense(layer_size=50, activation_func=ReLu(), weight_initializer=XavierInitializer()),
#             Dense(layer_size=10, activation_func=Softmax(), weight_initializer=XavierInitializer())
#         ]
#     },
#     {
#         'name': 'one_layer_10_neurons',
#         'Layer': [
#             Dense(layer_size=10, activation_func=ReLu(), weight_initializer=XavierInitializer()),
#             Dense(layer_size=10, activation_func=Softmax(), weight_initializer=XavierInitializer())
#         ]
#     },
#     {
#         'name': 'one_layer_28_neurons',
#         'Layer': [
#             Dense(layer_size=28, activation_func=ReLu(), weight_initializer=XavierInitializer()),
#             Dense(layer_size=10, activation_func=Softmax(), weight_initializer=XavierInitializer())
#         ]
#     },
#     {
#         'name': 'one_layer_300_neurons',
#         'Layer': [
#             Dense(layer_size=300, activation_func=ReLu(), weight_initializer=XavierInitializer()),
#             Dense(layer_size=10, activation_func=Softmax(), weight_initializer=XavierInitializer())
#         ]
#     },
#     {
#         'name': 'two_layers_100_50_neurons',
#         'Layer': [
#             Dense(layer_size=100, activation_func=ReLu(), weight_initializer=XavierInitializer()),
#             Dense(layer_size=50, activation_func=ReLu(), weight_initializer=XavierInitializer()),
#             Dense(layer_size=10, activation_func=Softmax(), weight_initializer=XavierInitializer())
#         ]
#     },
#     {
#         'name': 'two_layers_100_10_neurons',
#         'Layer': [
#             Dense(layer_size=100, activation_func=ReLu(), weight_initializer=XavierInitializer()),
#             Dense(layer_size=10, activation_func=ReLu(), weight_initializer=XavierInitializer()),
#             Dense(layer_size=10, activation_func=Softmax(), weight_initializer=XavierInitializer())
#         ]
#     },
# ])
#
# test_batch_sizes([50000, 2048, 1024, 100, 32, 1])
