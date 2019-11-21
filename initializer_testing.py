from Init.normal_initializer import NormalInit
from Init.xavier_initializer import XavierInit
from Tests.initializer_tests import perform_initializer_test

initializers = [
    XavierInit(gain=6),
    NormalInit(loc=0, scale=1, a=10)
]

perform_initializer_test(initializers)
