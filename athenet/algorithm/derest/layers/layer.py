from athenet.algorithm.derest.activation import count_activation
from athenet.algorithm.derest.derivative import count_derivative


class DerestLayer(object):

    def __init__(self, layer):
        self.layer = layer
        self.activations = None
        self.derivatives = None

    def count_activation(self, input):
        return count_activation(input, self.layer)

    def count_derivatives(self, output, input_shape):
        #TODO - nice check if activations are done
        return count_derivative(output, self.activations,
                                input_shape, self.layer)

    def count_derest(self, f):
        pass
