from Tests.optimizer_tests import perform_optimizer_test

from Optimizer.AdaDelta import AdaDelta
from Optimizer.AdaGrad import AdaGrad
from Optimizer.Adam import Adam
from Optimizer.GradientDescent import GradientDescent
from Optimizer.GradientMomentum import Optimizer

optimizers = [
    GradientDescent(),
    Adam(),
    Optimizer(),
    AdaGrad(),
    AdaDelta()
]

perform_optimizer_test(optimizers)
