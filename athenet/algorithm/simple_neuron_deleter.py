"""
    sender - SimplE Neuron DEleteR

    Place for some description
"""

import numpy as np

from athenet import Network
from athenet.layers import FullyConnectedLayer


def simple_neuron_deleter(network, p, layer_limit):
    """ (this docstring should be writen better)
        :p, layer_limit: - floats between 0 and 1. Deletes [p] neurons from
            layers connected direclty to fully connected layer's, but at most
            [layer_limit] neurons from single layer.
    """
    # collect information about layers

    def list_of_percentage_columns(layer_id, layer):
        W = np.transpose(layer.W)
        all_weights = np.sum(abs(W))
        result = []
        for i in xrange(W.shape[0]):
            result.append((np.sum(abs(W[i])) / all_weights, i, layer_id))
        return result

    def delete_column(layer, i):
        W = layer.W
        for j in xrange(W.shape[0]):
            W[j][i] = 0.
        layer.W = W

    all_columns = []
    n_layers = 0
    for i in xrange(len(network.weighted_layers)):
        layer = network.weighted_layers[i]
        if isinstance(layer, FullyConnectedLayer):
            all_columns += list_of_percentage_columns(i, layer)
            n_layers += 1

    all_columns = sorted(all_columns)

    deleted_for_layer = np.zeros((len(network, weighted_layers)))
    deleted_in_general = 0.
    for val, column, layer_id in all_columns:
        if deleted_in_general + val / n_layers > p:
            break
        if val + deleted_for_layer[layer_id] > layer_limit:
            continue
        deleted_for_layer[layer_id] += val
        delete_column(network.weighted_layers[layer_id], column)
        deleted_in_general += val / n_layers
