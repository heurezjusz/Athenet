import numpy as np


def set_zeros_on_layer(layer, zeroed_weights_fraction, order):
    """
    Change weights in layer to zeros.

    This function, for given order of weights,
    changes the given fraction of the smallest weights to zeros or,
    if that is not possible, takes the ceiling of such number.

    :param WeightedLayer layer: layer for sparsifying
    :param float zeroed_weights_fraction:
        fraction of weights to be changed to zeros
    :param order: order of weights
    """

    if zeroed_weights_fraction == 0:
        return

    W = layer.W
    percentile = np.percentile([order(a) for a in W.flat],
                               zeroed_weights_fraction * 100)
    W[order(W) <= percentile] = 0
    layer.W = W


def set_zeros_on_network(network, zeroed_weights_fraction, order):
    """
    Change weights in network to zeros.

    This function, for given order of weights,
    change the given fraction of the smallest to zeros or,
    if that is not possible, takes the ceiling of such number.

    :param Network network: network for sparsifying
    :param float zeroed_weights_fraction:
        fraction of weights to be changed to zeros
    :param order: order of weights
    :type order: function
    """
    if zeroed_weights_fraction == 0:
        return

    weights = np.concatenate(
        [layer.W.flatten() for layer in network.weighted_layers])
    percentile = np.percentile([order(a) for a in weights.flat],
                               zeroed_weights_fraction * 100)
    for layer in network.weighted_layers:
        W = layer.W
        W[order(W) <= percentile] = 0
        layer.W = W


def sparsify_smallest_on_network(network, zeroed_weights_fraction):
    """
    Change smallest weights in network to zeros.

    This function changes weights in such a way,
    that for the whole network the given fraction
    of the smallest of them are zeros,
    or, if that is not possible, takes the ceiling of such a number.

    :param Network network: network for sparsifying
    :param float fraction: percentage of weights to be changed to zeros
    """
    set_zeros_on_network(network, zeroed_weights_fraction, abs)


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

    weights = np.concatenate(
        [layer.W.flatten() for layer in network.weighted_layers])
    mean = np.mean(weights)
    set_zeros_on_network(network, zeroed_weights_fraction,
                         lambda x: abs(mean - x))


def sparsify_smallest_on_layers(network, zeroed_weights_fraction):
    """
    Change smallest weights in each layer to zeros.

    This function for each layer changes weights in such a way,
    that at least the given fraction of the smallest of them are zeros.

    :param Network network: network for sparsifying
    :param float zeroed_weights_fraction:
        fraction of weights to be changed to zeros
    """

    for layer in network.weighted_layers:
        set_zeros_on_layer(layer, zeroed_weights_fraction, abs)


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
    for layer in network.weighted_layers:
        mean = np.mean(layer.W.flatten())
        set_zeros_on_layer(layer, zeroed_weights_fraction,
                           lambda x: abs(mean - x))
