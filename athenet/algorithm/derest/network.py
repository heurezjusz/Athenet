import numpy

from athenet.algorithm.derest.utils import change_order, add_tuples,\
    make_iterable
from athenet.algorithm.derest.layers import DerestSoftmaxLayer,\
    DerestReluLayer, DerestPoolLayer, DerestInceptionLayer, DerestNormLayer,\
    DerestFullyConnectedLayer, DerestConvolutionalLayer, DerestDropoutLayer
from athenet.layers import Softmax, ReLU, PoolingLayer, LRN, \
    ConvolutionalLayer, Dropout, FullyConnectedLayer, InceptionLayer


def get_derest_layer(layer):
    """
    Return derest layer on which we can count activations, derivatives
        and derest algorithm

    :param Layer layer: network's original layer
    :return DerestLayer: new better derest layer
    """
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
        a = data.abs().amax().upper
        return data / a

    def count_activations(self, inp, normalize=False):
        """
        Computes estimated activations for each layer

        :param Numlike inp: input of network
        :param boolean normalize: whenever normalize number between layers
        :return Numlike: possible output for network
        """
        for layer in self.layers:
            if normalize:
                inp = self._normalize(inp)
            input_shape = change_order(make_iterable(layer.layer.input_shape))
            inp = inp.reshape(input_shape)
            layer.activations = inp
            inp = layer.count_activation(inp)
        return inp

    def count_derivatives(self, outp, normalize=False):
        """
        Computes estimated impact of input of each layer on output of network

        :param Numlike outp: output of network
        :param boolean normalize: whenever normalize number between layers
        :return Numlike:
        """
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
        """
        Returns indicators of each weight importance

        :param function count_function: function to count indicators,
            takes Numlike and returns float
        :return list of numpy arrays:
        """
        ""
        result = []
        for layer in self.layers:
            indicators = layer.count_derest(count_function)
            result.extend(indicators)
        return result
