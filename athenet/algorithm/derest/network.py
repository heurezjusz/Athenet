import numpy

from athenet.algorithm.derest.layers import get_derest_layer
from athenet.algorithm.derest.utils import change_order, add_tuples,\
    make_iterable


class DerestNetwork(object):

    def __init__(self, network):
        self.network = network
        self.layers = [get_derest_layer(layer)
                       for layer in network.layers]

    def count_activations(self, inp, normalize=False):
        """
        Computes estimated activations for each layer

        :param Numlike inp: input of network
        :param boolean normalize: whenever normalize number between layers
        :return Numlike: possible output for network
        """
        for layer in self.layers:
            inp = layer.count_activation(inp, normalize)
        return inp

    def count_derivatives(self, outp, normalize=False):
        """
        Computes estimated impact of input of each layer on output of network

        :param Numlike outp: output of network
        :param boolean normalize: whenever normalize number between layers
        :return Numlike:
        """
        batches = outp.shape[0]
        for layer in reversed(self.layers):
            outp = layer.count_derivatives(outp, batches, normalize)
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
