"""
    Deletes neurons from fully connected layers

    Place for some description
"""

import numpy as np

from athenet.layers import FullyConnectedLayer, ConvolutionalLayer, MaxPool
from athenet.algorithm.utils import list_of_percentage_columns, \
    list_of_percentage_rows, delete_column, delete_row


def middlelayer_neuron_deleter(network, config):
    """I think this algorithm is supposed to have better name.

        :config: - tuple of 2 foats, p, and layer_limit
        It do not support pooling and convolutional layers between
        fully connected layers
    """
    p, layer_limit = config

    assert p >= 0. and p <= 1.
    assert layer_limit >= 0. and layer_limit <= 1.

    weights = []
    was_fully_connected_layer = False

    neurons_for_layer = np.zeros((len(network.layers)))
    neurons_in_general = 0
    deleted_for_layer = np.zeros((len(network.layers)))
    deleted_in_general = 0

    for i in xrange(len(network.layers)):
        layer = network.layers[i]
        if isinstance(layer, FullyConnectedLayer):
            was_fully_connected_layer = True
            weights.append((list_of_percentage_rows(i, layer),
                            list_of_percentage_columns(i, layer)))
            neurons_for_layer[i] = layer.W.shape[0]
        elif was_fully_connected_layer:
            assert not isinstance(layer, ConvolutionalLayer)
            assert not isinstance(layer, MaxPool)

    all_neurons = []
    for i in xrange(len(weights) - 1):
        for j in xrange(len(weights[i][1])):
            # value, number of column, column_layer_id, row_layer_id
            neurons_in_general += neurons_for_layer[i]
            all_neurons.append((weights[i][1][j][0] * weights[i + 1][0][j][0],
                                weights[i][1][j][1], weights[i][1][j][2],
                                weights[i + 1][0][j][2]))

    all_neurons = sorted(all_neurons)
    for val, neuron_id, column_layer_id, row_layer_id in all_neurons:
        if deleted_in_general >= p * neurons_in_general:
            break
        if deleted_for_layer[row_layer_id] + 1 > \
                layer_limit * neurons_for_layer[row_layer_id]:
            continue
        delete_column(network.layers[column_layer_id], neuron_id)
        delete_row(network.layers[row_layer_id], neuron_id)
        deleted_for_layer[row_layer_id] += 1
        deleted_in_general += 1

