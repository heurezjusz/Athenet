"""
    sender - SimplE Neuron DEleteR

    Deletes the least significant neurons. Value of neuron is defined as
    sum of absolute values of weights outgoing from neuron divided by
    sum of absolute values of weights outgoing from entire layer.
"""

import numpy as np

from athenet.layers import FullyConnectedLayer
from athenet.algorithm.utils import list_of_percentage_rows, delete_row
from athenet.algorithm.deleting import delete_weights_by_global_fraction


def simple_neuron_indicators(layers, p, layer_limit):
    """
        Returns list of indicators.

        This function, for a given set of layers,
        computes importance indicators for every weight.
        Weights are considered in sets corresponding to neurons in network,
        which in fully connected layers are represented as rows if weight's
        matrix.

        If some weight is not going to be erased then its indicator
        is set to -1.

        :param layers: list of layers
        :type layers: list of instances of athenet.layers.FullyConnectedLayer
        :param p: float between 0 and 1, fraction of neurons to be considered
        :param layer_limit: float between 0 and 1, maximal fraction of neurons
                            which will be considered in a single layer.
    """
    assert p >= 0. and p <= 1.
    assert layer_limit >= 0. and layer_limit <= 1.
    if layer_limit < p:
        p = layer_limit
    for layer in layers:
        assert(isinstance(layer, FullyConnectedLayer))

    # counter of neurons
    neurons_for_layer = np.zeros((len(layers),))
    neurons_in_general = 0
    # counter of deleted neurons
    deleted_for_layer = np.zeros((len(layers),))
    deleted_in_general = 0

    # results
    results = []

    # list of all neurons (interpreted as rows of matrices)
    considered_neurons = []
    for i in xrange(len(layers)):
        layer = layers[i]
        considered_neurons += list_of_percentage_rows(i, layer)
        neurons_for_layer[i] = layer.W.shape[0]
        neurons_in_general += neurons_for_layer[i]
        results.append(-np.ones_like(layer.W))

    considered_neurons = sorted(considered_neurons)

    for val, row, layer_id in considered_neurons:
        if deleted_in_general >= p * neurons_in_general:
            break
        if 1 + deleted_for_layer[layer_id] > \
                layer_limit * neurons_for_layer[i]:
            continue
        deleted_for_layer[layer_id] += 1
        results[layer_id][row] = val
        deleted_in_general += 1

    return results


def simple_neuron_deleter(network, p, layer_limit):
    """
        :param network: an instance of athenet.Network.
        :param p: float between 0 and 1, fraction of neurons to be deleted
                  from fully connected layers
        :param layer_limit: float between 0 and 1, maximal fraction of neurons
                            which will be deleted from a single layer.

        Modifies [network]. Deletes [p] neurons from layers connected direclty
        to fully connected layers. Do not delete more than [layer_limit]
        neurons from a single layer.
        If [layer_limit] < [p] then at most [layer_limit] neurons will be
        deleted.

        Deletion of neuron is simulated by replacing all weights outgoing
        form to it by 0. In athenet.network they are reprezented as rows
        of next layer's weights matrix.
    """
    fully_connected_layers = [layer for layer in network.weighted_layers
                              if isinstance(layer, FullyConnectedLayer)]
    indicators = simple_neuron_indicators(fully_connected_layers, p,
                                          layer_limit)
    delete_weights_by_global_fraction(fully_connected_layers, p, indicators)
