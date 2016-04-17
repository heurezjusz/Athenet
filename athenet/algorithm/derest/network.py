import theano

from athenet.layers import ConvolutionalLayer, PoolingLayer, FullyConnectedLayer, Softmax, Dropout, ReLU, LRN
from athenet.algorithm.derest.activation import *
from athenet.algorithm.derest.derivative import *

def _change_order(a):
    """
    So the last will be first
    """
    try:
        h, w, n = a
        return (n, h, w)
    except:
        return a


def _add_tuples(a, b):
    if not isinstance(a, tuple):
        a = (a, )
    if not isinstance(b, tuple):
        b = (b, )
    return a + b


class DerestNetwork():

    def __init__(self, network):
        self.network = network
        self.layers = [DerestLayer(layer) for layer in network.layers]

    def _get_layer_input_shape(self, i):
        if i > 0:
            return self.layers[i - 1].layer.output_shape
        return self.layers[i].layer.input_shape

    def count_activations(self, inp):
        for layer in self.layers:
            inp = layer.count_activation(inp)
        return inp

    def count_derivatives(self, outp, batches):
        #we assume that batches is equal to outp.shape[0] (for now)
        for i in range(len(self.layers) - 1, -1, -1):
            input_shape = _add_tuples(batches, _change_order(self._get_layer_input_shape(i)))
            outp = self.layers[i].count_derivatives(outp, input_shape)
        return outp


class DerestLayer():

    def __init__(self, layer):
        self.layer = layer
        self.activations = None
        self.derivatives = None

    def count_activation(self, input):
        self.activations = input
        if isinstance(self.layer, ConvolutionalLayer):
            return a_conv(
                input, _change_order(self.layer.input_shape),
                self.layer.W, _change_order(self.layer.filter_shape),
                theano.shared(self.layer.b), self.layer.stride, self.layer.padding
            )
        elif isinstance(self.layer, Dropout):
            return a_dropout(input, self.layer.p_dropout)
        elif isinstance(self.layer, FullyConnectedLayer):
            return a_fully_connected(input, self.layer.W,
                                               self.layer.b)
        elif isinstance(self.layer, LRN):
            return a_norm(
                input, _change_order(self.layer.input_shape),
                self.layer.local_range, self.layer.k,
                self.layer.alpha, self.layer.beta
            )
        elif isinstance(self.layer, PoolingLayer):
            return a_pool(
                input, _change_order(self.layer.input_shape),
                self.layer.poolsize, self.layer.stride, self.layer.mode
            )
        elif isinstance(self.layer, Softmax):
            return a_softmax(input, self.layer.input_shape)
        elif isinstance(self.layer, ReLU):
            return a_relu(input)
        else:
            raise NotImplementedError


    def count_derivatives(self, output, input_shape):
        if isinstance(self.layer, ConvolutionalLayer):
            self.derivatives = d_conv(
                output, input_shape,
                _change_order(self.layer.filter_shape), self.layer.W,
                self.layer.stride, self.layer.padding, self.layer.n_groups
            )
        elif isinstance(self.layer, Dropout):
            self.derivatives = d_dropout(output, self.layer.p_dropout)
        elif isinstance(self.layer, FullyConnectedLayer):
            self.derivatives = d_fully_connected(output, self.layer.W,
                                                 input_shape)
        elif isinstance(self.layer, LRN):
            self.derivatives = d_norm(
                output, self.activations, input_shape,
                self.layer.local_range, self.layer.k, self.layer.alpha,
                self.layer.beta
            )
        elif isinstance(self.layer, PoolingLayer):
            self.derivatives = d_pool(
                output, self.activations, input_shape,
                self.layer.poolsize, self.layer.stride, self.layer.padding,
                self.layer.mode
            )
        elif isinstance(self.layer, Softmax):
            self.derivatives = d_softmax(output)
        elif isinstance(self.layer, ReLU):
            self.derivatives = d_relu(output, self.activations)
        else:
            raise NotImplementedError

        return self.derivatives
