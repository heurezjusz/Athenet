import numpy
from athenet.tests.mock_network import LayerMock, NetworkMock


def get_fraction_of_zeros_in_layer(layer):
    return 1. - numpy.count_nonzero(layer.W) / float(layer.W.size)


def get_fraction_of_zeros_in_network(network):
    number_of_nonzeros = sum((numpy.count_nonzero(layer.W)
                              for layer in network.weighted_layers))
    number_of_weights = sum((layer.W.size
                            for layer in network.weighted_layers))
    return 1. - (number_of_nonzeros / float(number_of_weights))


def get_random_layer_mock(shape_of_layer=100):
    return LayerMock(
        weights=numpy.random.uniform(low=-1, high=1, size=shape_of_layer),
        biases=numpy.random.uniform(low=-1, high=1, size=shape_of_layer))


def get_random_network_mock(number_of_layers=5, shape_of_layer=100):
    layers = [get_random_layer_mock(shape_of_layer=shape_of_layer)
              for x in xrange(number_of_layers)]
    return NetworkMock(weighted_layers=layers)
