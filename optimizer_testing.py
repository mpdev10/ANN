from Optimizer.adadelta_optimizer import AdaDelta
from Optimizer.adagrad_optimizer import AdaGrad
from Optimizer.adam_optimizer import Adam
from Optimizer.gradient_descent_static import GradientDescent
from Optimizer.momentum_optimizer import GradientMomentum
from Tests.optimizer_tests import perform_optimizer_test

optimizers = [
    GradientDescent(),
    Adam(),
    GradientMomentum(),
    AdaGrad(),
    AdaDelta()
]

perform_optimizer_test(optimizers)
