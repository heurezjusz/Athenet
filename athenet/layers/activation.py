"""Activation layers."""

import theano.tensor as T

from athenet.layers import Layer


class ActivationLayer(Layer):
    """Layer applying activation function to neurons."""
    def __init__(self, activation_function, input_layer_name=None, name='act'):
        """Create activation layer.

        :param activation_function: Activation function to be applied.
        """
        super(ActivationLayer, self).__init__(input_layer_name, name)
        self.activation_function = activation_function

    def _get_output(self, layer_input):
        return self.activation_function(layer_input)


def relu(x):
    """Rectified linear activation function.

    :param x: Neuron input.
    """
    return T.maximum(0., x)


class ReLU(ActivationLayer):
    """Layer applying rectified linear activation function."""
    def __init__(self, input_layer_name=None, name='relu'):
        super(ReLU, self).__init__(relu, input_layer_name, name)
