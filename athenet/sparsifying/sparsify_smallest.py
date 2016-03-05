import numpy as np


def set_zeros_on_layer(layer, percentage, order):
    """
    Change weights in layer to zeros.

    This function, for given order of weights,
    changes the given percentage of the smallest weights to zeros or,
    if that is not possible, takes the ceiling of such number.

    :param WeightedLayer layer: layer for sparsifying
    :param percentage: percentage of weights to be changed to zeros
    :param order: order of weights
    :type percentage: int, float
    """

    if percentage == 0:
        return

    W = layer.W
    percentile = np.percentile([order(a) for a in W.flat], percentage)
    W[order(W) <= percentile] = 0
    layer.W = W


def set_zeros_on_network(network, percentage, order):
    """
    Change weights in network to zeros.

    This function, for given order of weights,
    change the given percentage of the smallest to zeros or,
    if that is not possible, takes the ceiling of such number.

    :param Network network: network for sparsifying
    :param percentage: percentage of weights to be changed to zeros
    :param order: order of weights
    :type percentage: int, float
    :type order: function
    """
    if percentage == 0:
        return

    weights = np.concatenate(
        [layer.W.flatten() for layer in network.weighted_layers])
    percentile = np.percentile([order(a) for a in weights.flat], percentage)
    for layer in network.weighted_layers:
        W = layer.W
        W[order(W) <= percentile] = 0
        layer.W = W


def sparsify_smallest_on_network(network, percentage):
    """
    Change smallest weights in network to zeros.

    This function changes weights in such a way,
    that for the whole network the given percentage
    of the smallest of them are zeros,
    or, if that is not possible, takes the ceiling of such a number.

    :param Network network: network for sparsifying
    :param percentage: percentage of weights to be changed to zeros
    :type percentage: int, float
    """
    set_zeros_on_network(network, percentage, abs)


def sparsify_nearest_to_network_mean(network, percentage):
    """
    Change weights close to network's mean to zeros.

    This function changes weights in such a way,
    that for the whole network the given percentage
    of the closest of them to mean are zeros,
    or, if that is not possible, takes the ceiling of such a number.

    :param Network network: network for sparsifying
    :param percentage: percentage of weights to be changes to zeros
    :type percentage: int, float
    """

    weights = np.concatenate(
        [layer.W.flatten() for layer in network.weighted_layers])
    mean = np.mean(weights)
    set_zeros_on_network(network, percentage, lambda x: abs(mean - x))


def sparsify_smallest_on_layers(network, percentage):
    """
    Change smallest weights in each layer to zeros.

    This function for each layer changes weights in such a way,
    that at least the given percentage of the smallest of them are zeros.

    :param Network network: network for sparsifying
    :param percentage: percentage of weights to be changed to zeros
    :type percentage: int, float
    """

    for layer in network.weighted_layers:
        set_zeros_on_layer(layer, percentage, abs)


def sparsify_nearest_to_layer_mean(network, percentage):
    """
    In each layer, change weights closest to mean to zeros.

    This function for each layer changes weights in such a way,
    that the given percentage of the closest to mean are set to zero
    or, if that is not possible, takes the ceiling of such a number.

    :param Network network: network for sparsifying
    :param percentage: percentage of weights to be changed to zeros
    :type percentage: int, float
    """
    for layer in network.weighted_layers:
        mean = np.mean(layer.W.flatten())
        set_zeros_on_layer(layer, percentage, lambda x: abs(mean - x))
