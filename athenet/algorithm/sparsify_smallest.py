import numpy

from athenet.algorithm.utils import set_zeros_by_layer_fractions,\
    set_zeros_by_global_fraction


def get_smallest_indicators(layers):
    return numpy.array([1 - abs(layer.W) for layer in layers])


def get_nearest_to_global_mean_indicators(layers):
    weights = numpy.concatenate(
        [layer.W.flatten() for layer in layers])
    mean = numpy.mean(weights)
    return numpy.array([abs(mean - layer.W) for layer in layers])


def get_nearest_to_layer_mean_indicators(layer):
    mean = numpy.mean(layer.W)
    return abs(mean - layer.W)


def get_nearest_to_layers_mean_indicators(layers):
    return numpy.array([get_nearest_to_layer_mean_indicators(layer)
                        for layer in layers])


def sparsify_smallest_on_network(network, zeroed_weights_fraction):
    """
    Change smallest weights in network to zeros.

    This function changes weights in such a way,
    that for the whole network the given fraction
    of the smallest of them are zeros,
    or, if that is not possible, takes the ceiling of such a number.

    :param Network network: network for sparsifying
    :param float zeroed_weights_fraction:
        percentage of weights to be changed to zeros
    """
    indicators = get_smallest_indicators(network.weighted_layers)
    set_zeros_by_global_fraction(network.weighted_layers,
                                 zeroed_weights_fraction, indicators)


def sparsify_nearest_to_network_mean(network, zeroed_weights_fraction):
    """
    Change weights close to network's mean to zeros.

    This function changes weights in such a way,
    that for the whole network the given fraction
    of the closest of them to mean are zeros,
    or, if that is not possible, takes the ceiling of such a number.

    :param Network network: network for sparsifying
    :param float zeroed_weights_fraction:
        fraction of weights to be changes to zeros
    """
    indicators = get_nearest_to_global_mean_indicators(
        network.weighted_layers)
    set_zeros_by_global_fraction(network.weighted_layers,
                                 zeroed_weights_fraction, indicators)


def sparsify_smallest_on_layers(network, zeroed_weights_fraction):
    """
    Change smallest weights in each layer to zeros.

    This function for each layer changes weights in such a way,
    that at least the given fraction of the smallest of them are zeros.

    :param Network network: network for sparsifying
    :param float zeroed_weights_fraction:
        fraction of weights to be changed to zeros
    """
    layers = network.weighted_layers
    indicators = get_smallest_indicators(layers)
    fractions = numpy.ones(len(layers)) * zeroed_weights_fraction
    set_zeros_by_layer_fractions(layers, fractions, indicators)


def sparsify_nearest_to_layer_mean(network, zeroed_weights_fraction):
    """
    In each layer, change weights closest to mean to zeros.

    This function for each layer changes weights in such a way,
    that the given fraction of the closest to mean are set to zero
    or, if that is not possible, takes the ceiling of such a number.

    :param Network network: network for sparsifying
    :param float zeroed_weights_fraction:
        fraction of weights to be changed to zeros
    """
    layers = network.weighted_layers
    indicators = get_nearest_to_layers_mean_indicators(layers)
    fractions = numpy.ones(len(layers)) * zeroed_weights_fraction
    set_zeros_by_layer_fractions(layers, fractions, indicators)
