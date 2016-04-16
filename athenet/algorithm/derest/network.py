from athenet.network import Network
from athenet.layers import *
from athenet.algorithm.derest.activation import *
from athenet.algorithm.derest.derivative import *


def _change_order((h, w, n)):
    """
    So the last will be first, and the first will be last.
    """
    return (n, h, w)


class derestNetwork(Network):

    def __init__(self, network):
        self.network = network


class derestLayer(Layer):

    def __init__(self, layer):
        self.layer = layer
        self.layer_input = None
        self.activations = None
        self.derivatives = None

        self._choose_functions()


    def count_activation(self, input):
        if isinstance(self.layer, ConvolutionalLayer):
            self.activations = conv(
                input, _change_order(self.layer.input_shape),
                self.layer.W, _change_order(self.layer.filter_shape),
                self.layer.b, self.layer.stride, self.layer.padding
            )
        elif isinstance(self.layer, Dropout):
            self.activations = dropout(input, self.layer.p_dropout)
        elif isinstance(self.layer, FullyConnectedLayer):
            self.activations = fully_connected(input, self.layer.W,
                                               self.layer.b)
        elif isinstance(self.layer, LRN):
            self.activations = norm(
                input, _change_order(self.layer.input_shape),
                self.layer.local_range, self.layer.k,
                self.layer.alpha, self.layer.beta
            )
        elif isinstance(self.layer, PoolingLayer):
            self.activations = pool(
                input, _change_order(self.layer.input_shape),
                self.layer.poolsize, self.layer.stride, self.layer.mode
            )
        elif isinstance(self.layer, Softmax):
            self.activations = softmax(input, self.layer.input_shape)
        elif isinstance(self.layer, ReLU):
            self.activations = relu(input)
        else:
            raise NotImplementedError


    def count_derivatives(self, output):
        if isinstance(self.layer, ConvolutionalLayer):
            self.derivatives = d_conv(
                output, self.activations.shape,
                _change_order(self.layer.filter_shape), self.layer.W,
                self.layer.stride, self.layer.padding, self.layer.n_groups
            )
        elif isinstance(self.layer, Dropout):
            self.derivatives = d_dropout(output, self.layer.p_dropout)
        elif isinstance(self.layer, FullyConnectedLayer):
            self.derivatives = d_fully_connected(output, self.layer.W,
                                                 self.layer.input_shape)
        elif isinstance(self.layer, LRN):
            self.derivatives = d_norm(
                output, self.activations, self.activations.shape,
                self.layer.local_range, self.layer.k, self.layer.alpha,
                self.layer.beta
            )
        elif isinstance(self.layer, PoolingLayer):
            self.derivatives = d_pool(
                output, self.activations, self.activations.shape,
                self.layer.poolsize, self.layer.stride, self.layer.padding,
                self.layer.mode
            )
        elif isinstance(self.layer, Softmax):
            return d_softmax(output)
        elif isinstance(self.layer, ReLU):
            return d_relu(output, self.activations)
        else:
            raise NotImplementedError




