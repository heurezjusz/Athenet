import numpy


def _delete_weights_in_layer_by_fraction(layer, fraction,
                                         importance_indicator):
    if fraction == 0:
        return

    W = layer.W
    percentile = numpy.percentile(importance_indicator, (1 - fraction) * 100)
    W[importance_indicator >= percentile] = 0
    layer.W = W


def delete_weights_by_layer_fractions(layers, fractions,
                                      importance_indicators):
    """
    Change weights in layer to zeros.

    This function, for given order of weights,
    changes the given fraction of the smallest weights to zeros or,
    if that is not possible, takes the ceiling of such number.

    :param WeightedLayer layer: layer for sparsifying
    :param float or list or numpy.array zeroed_weights_fraction:
        fraction of weights to be changed to zeros
    :param order: order of weights
    """

    try:
        iter(fractions)
    except TypeError:
        fractions = [fractions for i in layers]

    for layer, fraction, importance_indicator \
            in zip(layers, fractions, importance_indicators):
        _delete_weights_in_layer_by_fraction(layer, fraction,
                                             importance_indicator)


def delete_weights_by_global_fraction(layers, zeroed_weights_fraction,
                                      importance_indicators):
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

    flattened_importance_indicators = numpy.concatenate(
        [layer_importance_indicators.flatten()
         for layer_importance_indicators in importance_indicators])

    percentile = numpy.percentile(flattened_importance_indicators,
                                  (1 - zeroed_weights_fraction) * 100)

    for layer, ord in zip(layers, importance_indicators):
        weights = layer.W
        weights[ord >= percentile] = 0
        layer.W = weights
