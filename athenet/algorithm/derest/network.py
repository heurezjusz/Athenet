import numpy

from athenet.algorithm.derest.utils import change_order, add_tuples
from athenet.algorithm.derest.layers import *
from athenet.layers import *

# TODO - add normalization of inputs and outputs between layers in count_activations and count_derivatives


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

    def _get_layer_input_shape(self, i):
        # TODO - do it better
        if i > 0:
            return self.layers[i - 1].layer.output_shape
        return self.layers[i].layer.input_shape

    @staticmethod
    def _normalize(data):
        a = max(numpy.abs(data.amax().eval()))
        return data / a

    def count_activations(self, inp, normalize=False):
        for layer in self.layers:
            if normalize:
                inp = self._normalize(inp)
            layer.activations = inp
            inp = layer.count_activation(inp)
        return inp

    def count_derivatives(self, outp, normalize=False):
        batches = outp.shape.eval()[0]
        for i in range(len(self.layers) - 1, -1, -1):
            if normalize:
                outp = self._normalize(outp)
            input_shape = add_tuples(
                batches,
                change_order(self._get_layer_input_shape(i))
            )
            self.layers[i].derivatives = outp
            outp = self.layers[i].count_derivatives(
                outp,
                input_shape
            )
        return outp

    def count_derest(self, f):
        result = []
        for layer in self.layers:
            indicators = layer.count_derest(f)
            result.extend(indicators)
        return result
