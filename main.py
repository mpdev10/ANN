from Init.NormalInit import NormalInit
from Init.XavierInit import XavierInit
from Tests.initializer_tests import perform_initializer_test

if __name__ == '__main__':
    xavierInit = XavierInit()
    normalInit = NormalInit(loc=0, scale=1, a=10)

    perform_initializer_test([xavierInit, normalInit])
