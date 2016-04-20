"""Functions that for any neuron and given range of inputs of layers calculate
neuron's range of activation. Functions are being invoked from the beginning
to the end of the network.
"""
import theano

from athenet.layers import ConvolutionalLayer, FullyConnectedLayer, \
    InceptionLayer, Dropout, LRN, PoolingLayer, Softmax, ReLU
from athenet.algorithm.numlike import Numlike, assert_numlike
from athenet.algorithm.derest.utils import _change_order


def count_activation(layer_input, layer):
    if isinstance(layer, LRN):
        return a_norm(
            layer_input, _change_order(layer.input_shape),
            layer.local_range, layer.k,
            layer.alpha, layer.beta
        )
    elif isinstance(layer, PoolingLayer):
        return a_pool(
            layer_input, _change_order(layer.input_shape),
            layer.poolsize, layer.stride, layer.mode
        )
    elif isinstance(layer, Softmax):
        return a_softmax(layer_input, layer.input_shape)
    elif isinstance(layer, ReLU):
        return a_relu(layer_input)
    else:
        raise NotImplementedError






