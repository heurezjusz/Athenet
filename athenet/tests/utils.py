import numpy


def get_fraction_of_zeros_in_layer(self, layer):
    return 1. - numpy.count_nonzero(layer.W.flat) / float(len(layer.W.flat))


def get_fraction_of_zeros_in_network(self, network):
    number_of_nonzeros = sum((numpy.count_nonzero(layer.W.flat)
                              for layer in network.weighted_layers))
    number_of_weights = sum((len(layer.W.flat)
                            for layer in network.weighted_layers))
    return 1. - (number_of_nonzeros / float(number_of_weights))
