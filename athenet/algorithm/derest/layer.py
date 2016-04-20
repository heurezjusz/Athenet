from athenet.layers import FullyConnectedLayer, ConvolutionalLayer, InceptionLayer
from athenet.algorithm.derest import DerestFullyConnectedLayer, DerestConvolutionalLayer, DerestInceptionLayer

from athenet.algorithm.derest.activation import count_activation
from athenet.algorithm.derest.derivative import count_derivative


def derest_layer(layer):
    if isinstance(layer, FullyConnectedLayer):
        return DerestFullyConnectedLayer(layer)
    elif isinstance(layer, ConvolutionalLayer):
        return DerestConvolutionalLayer(layer)
    elif isinstance(layer, InceptionLayer):
        return DerestInceptionLayer(layer)
    else:
        return DerestLayer(layer)


class DerestLayer():

    def __init__(self, layer):
        self.layer = layer
        self.activations = None
        self.derivatives = None

    def count_activation(self, input):
        return count_activation(input, self.layer)

    def count_derivatives(self, output, input_shape):
        self.derivatives = output
        return count_derivative(output, self.activations,
                                input_shape, self.layer)

    def count_derest(self, f):
        pass