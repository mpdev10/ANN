class Activation(object):
    name = 'activation'

    def compute(self, z):
        raise NotImplementedError

    def compute_deriv(self, a):
        raise NotImplementedError

    def get_name(self):
        return 'none'
