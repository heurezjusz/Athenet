import numpy

from athenet.algorithm.derest.utils import change_order, add_tuples, make_iterable
from athenet.algorithm.derest.layers import *
from athenet.layers import *


def get_derest_layer(layer):
    if isinstance(layer, Softmax):
        return DerestSoftmaxLayer(layer)
    if isinstance(layer, ReLU):
        return DerestReluLayer(layer)
    if isinstance(layer, PoolingLayer):
        return DerestPoolLayer(layer)
    if isinstance(layer, LRN):
        return DerestNormLayer(layer)
    if isinstance(layer, ConvolutionalLayer):
        return DerestConvolutionalLayer(layer)
    if isinstance(layer, Dropout):
        return DerestDropoutLayer(layer)
    if isinstance(layer, FullyConnectedLayer):
        return DerestFullyConnectedLayer(layer)
    if isinstance(layer, InceptionLayer):
        return DerestInceptionLayer(layer)
    raise NotImplementedError


class DerestNetwork(object):

    def __init__(self, network):
        self.network = network
        self.layers = [get_derest_layer(layer)
                       for layer in network.layers]

    @staticmethod
    def _normalize(data):
        a = max(numpy.abs(data.amax().eval()))
        return data / a

    def count_activations(self, inp, normalize=False):
        for layer in self.layers:
            if normalize:
                inp = self._normalize(inp)
            input_shape = change_order(make_iterable(layer.layer.input_shape))
            inp = inp.reshape(input_shape)
            layer.activations = inp
            inp = layer.count_activation(inp)
        return inp

    def count_derivatives(self, outp, normalize=False):
        batches = outp.shape.eval()[0]
        for layer in reversed(self.layers):
            if normalize:
                outp = self._normalize(outp)
            input_shape = add_tuples(batches,
                                     change_order(layer.layer.input_shape))
            output_shape = add_tuples(batches,
                                      change_order(layer.layer.output_shape))
            outp = outp.reshape(output_shape)
            layer.derivatives = outp
            outp = layer.count_derivatives(outp, input_shape)
        return outp

    def count_derest(self, count_function):
        result = []
        for layer in self.layers:
            indicators = layer.count_derest(count_function)
            result.extend(indicators)
        return result
