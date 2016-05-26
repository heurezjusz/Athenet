import numpy

from athenet.algorithm import get_derest_indicators,\
    delete_weights_by_global_fraction, get_smallest_indicators
from athenet.network import Network
from athenet.layers import ConvolutionalLayer, FullyConnectedLayer,\
    Softmax, ReLU, MaxPool, InceptionLayer, LRN


def custom_derivatives_normalization(data):
    return data / 4.


def custom_activations_normalization(data):
    return data / (data.sum().upper + 0.5)


def custom_count_function(data):
    return numpy.prod(data.upper) - numpy.prod(data.lower)


n = Network([
    ConvolutionalLayer(image_shape=(28, 28, 1), filter_shape=(4, 4, 2)),
    ReLU(),
    LRN(),
    MaxPool(poolsize=(2, 2)),
    InceptionLayer(n_filters=[2, 2, 2, 2, 2, 2]),
    FullyConnectedLayer(n_out=10),
    ReLU(),
    FullyConnectedLayer(n_out=3),
    Softmax(),
])

ind_derest = get_derest_indicators(
    n, max_batch_size=None, count_function=custom_count_function,
    normalize_activations=custom_activations_normalization,
    normalize_derivatives=custom_derivatives_normalization)

ind_rat = get_smallest_indicators(n.weighted_layers)
ind = [derest * rat for derest, rat in zip(ind_derest, ind_rat)]

delete_weights_by_global_fraction(n.weighted_layers, 0.6, ind)
