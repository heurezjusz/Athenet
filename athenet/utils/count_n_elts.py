"""Function that for given neural network, returns lists of numbers of edges,
weights and neurons per layer.
"""

from athenet.models import lenet, alexnet
from athenet.layers import FullyConnectedLayer, ConvolutionalLayer, \
        ActivationLayer, Dropout, Softmax, MaxPool, LRN
import numpy as np


def _conv_edges(x, y, fx, fy, sx, sy, n_in, n_out, g=1):
    """Number of edges in convolution."""
    nx = (x - fx) / sx + 1
    ny = (y - fy) / sy + 1
    return (fx * fy * nx * ny * n_in * n_out) / g


def _max_pool_edges(x, y, fx, fy, sx, sy, n_channels):
    """Number of edges in max pool."""
    return _conv_edges(x, y, fx, fy, sx, sy, n_channels, 1)


def _lrn_edges(x, y, n_channel, lr):
    """Number of edges in lrn."""
    return x * y * (n_channel * lr - (lr ** 2 - 1) / 4)


def count_statistics(network):
    """For given neural network, returns lists of numbers of edges, weights and
    neurons per layer.

    :param network: neural network in which edges, weights and layers will be
                    counted.

    For n layers, list of edges and weights are of length n and list of neurons
    is of length (n + 1). List of edges tells how many connections are between
    inputs and outputs of layers. List of weights tells how many elements are
    in weight matrices in layers. List of neurons tells how many elements are
    in inputs/outputs of the layers.

    Note: Number of edges in softmax for input size N is N in this calculation.
    It comes from the fact that softmax need O(N) operations to calculate
    output from input."""
    layers = network.layers
    n_edges, n_weights, n_neurons = [], [], []
    n_neurons += [np.prod(layers[0].input_shape)]
    for layer in layers:
        n_neurons += [np.prod(layer.output_shape)]
        if isinstance(layer, FullyConnectedLayer):
            n_weights += [np.prod(layer.W.shape)]
        elif isinstance(layer, ConvolutionalLayer):
            n_weights += [np.prod(layer.W.shape)]
        else:
            n_weights += [0]
        if isinstance(layer, ConvolutionalLayer):
            x, y, n_in = layer.image_shape
            fx, fy, n_out = layer.filter_shape
            sx, sy = layer.stride
            g = layer.n_groups
            n_edges += [_conv_edges(x, y, fx, fy, sx, sy, n_in, n_out, g)]
            continue
        if isinstance(layer, FullyConnectedLayer):
            n_edges += [layer.n_in * layer.n_out]
            continue
        if isinstance(layer, (ActivationLayer, Dropout, Softmax)):
            n_edges += [np.prod(layer.input_shape)]
            continue
        if isinstance(layer, MaxPool):
            fx, fy = layer.poolsize
            sx, sy = layer.stride
            x, y, n_channels = layer.input_shape
            n_edges += [_max_pool_edges(x, y, fx, fy, sx, sy, n_channels)]
            continue
        if isinstance(layer, LRN):
            x, y, n_channel = layer.input_shape
            lr = layer.local_range
            n_edges += [_lrn_edges(x, y, n_channel, lr)]
            continue
        raise ValueError("illegal layer type for counting elts")
    return (n_edges, n_weights, n_neurons)


if __name__ == "__main__":
    """Count number of edges, weights and neurons in LeNet and AlexNet"""
    print """LeNet"""
    lenet_network = lenet(trained=False)
    edges, weights, neurons = count_statistics(lenet_network)
    for l, edge, weight, neuron in zip(lenet_network.layers, edges, weights,
                                       neurons):
        print type(l), l.output_shape
        print 'edges:', edge, 'weights:', weight, 'neurons:', neuron
    print 'sum of edges:', sum(edges)
    print 'sum of weigths:', sum(weights)
    print 'sum of neurons:', sum(neurons)

    print """AlexNet"""
    alexnet_network = alexnet(trained=False)
    edges, weights, neurons = count_statistics(alexnet_network)
    for l, edge, weight, neuron in zip(alexnet_network.layers, edges, weights,
                                       neurons):
        print type(l), l.output_shape
        print 'edges:', edge, 'weights:', weight, 'neurons:', neuron
    print 'sum of edges:', sum(edges)
    print 'sum of weigths:', sum(weights)
    print 'sum of neurons:', sum(neurons)
