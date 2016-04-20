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




def a_norm(layer_input, input_shape, local_range=5, k=1, alpha=0.0001,
           beta=0.75):
    """Returns estimated activation of LRN layer.

    :param Numlike layer_input: Numlike input
    :param input_shape: shape of Interval in format (n_channels, height, width)
    :param integer local_range: size of local range in local range
                                normalization
    :param integer k: local range normalization k argument
    :param integer alpha: local range normalization alpha argument
    :param integer beta: local range normalization beta argument
    :type input_shape: tuple of 3 integers
    :rtype: Numlike
    """
    assert_numlike(layer_input)
    try:
        return layer_input.op_norm(input_shape, local_range, k, alpha, beta)
    except NotImplementedError:
        half = local_range / 2
        sq = layer_input.square()
        n_channels, h, w = input_shape
        extra_channels = layer_input.from_shape((n_channels + 2 * half, h, w),
                                                neutral=True)
        extra_channels[half:half + n_channels, :, :] = sq
        local_sums = layer_input.from_shape(input_shape, neutral=True)

        for i in xrange(local_range):
            local_sums += extra_channels[i:i + n_channels, :, :]

        return layer_input / ((
            local_sums * (alpha / local_range) + k).power(beta))



