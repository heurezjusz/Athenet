import numpy as np


def list_of_percentage_rows_table(table, layer_id):
    """returns list of tuples (percentage, number of row, [layer_id])
       representing rows of [table]

       "percentage" is a sum of absolute values in row divided by
       sum of absolute values in all table

       result list is sorted by "number of row"
    """
    sum_of_all = np.sum(abs(table))
    result = []
    for i in xrange(table.shape[0]):
        result.append((np.sum(abs(table[i])) / sum_of_all, i, layer_id))
    return result


def list_of_percentage_columns(layer_id, layer):
    return list_of_percentage_rows_table(np.transpose(layer.W), layer_id)


def list_of_percentage_rows(layer_id, layer):
    return list_of_percentage_rows_table(layer.W, layer_id)


def delete_column(layer, i):
    W = layer.W
    W[:, i] = 0.
    layer.W = W


def delete_row(layer, i):
    W = layer.W
    W[i] = 0.
    layer.W = W


def set_zeros_by_layer_fraction(layer, fraction, importance_indicator):
    if fraction == 0:
        return

    W = layer.W
    percentile = np.percentile(importance_indicator, (1 - fraction) * 100)
    W[importance_indicator >= percentile] = 0
    layer.W = W

def set_zeros_by_layer_fractions(layers, fractions, importance_indicators):
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

    #assert correctness

    for layer, fraction, importance_indicator in zip(layers, fractions, importance_indicators):
        set_zeros_by_layer_fraction(layer, fraction, importance_indicator)


def set_zeros_by_global_fraction(layers, zeroed_weights_fraction, importance_indicators):
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

    flattened_importance_indicators = np.concatenate(
        [layer_importance_indicators.flatten()
         for layer_importance_indicators in importance_indicators])

    percentile = np.percentile(flattened_importance_indicators,
                               (1 - zeroed_weights_fraction) * 100)

    for layer, ord in zip(layers, importance_indicators):
        weights = layer.W
        weights[ord >= percentile] = 0
        layer.W = weights