"""Functions that for any neuron calculate its range of derivative of output
with respect to this neuron. Functions should be invoked from the end to the
beginning of the network.

Every estimated impact of tensor on output of network is stored with batches.
Every entity in batches store impact on different output of network.
"""

from athenet.layers import ConvolutionalLayer, FullyConnectedLayer, \
    InceptionLayer, Dropout, LRN, PoolingLayer, Softmax, ReLU
from athenet.algorithm.numlike import Numlike, assert_numlike
from athenet.algorithm.derest.utils import _change_order

# TODO: All functions below will be implemented.


def count_derivative(layer_output, activations, input_shape, layer):
    if isinstance(layer, LRN):
        return d_norm(
            layer_output, activations, input_shape,
            layer.local_range, layer.k, layer.alpha,
            layer.beta
        )
    elif isinstance(layer, PoolingLayer):
        return d_pool(
            layer_output, activations, input_shape,
            layer.poolsize, layer.stride, layer.padding,
            layer.mode
        )
    elif isinstance(layer, Softmax):
        return d_softmax(layer_output)
    elif isinstance(layer, ReLU):
        return d_relu(layer_output, activations)
    else:
        raise NotImplementedError



def d_norm(output, activation, input_shape, local_range, k, alpha, beta):
    # TODO: all
    """Returns estimated impact of input of LRN layer on output of network.

    :param Numlike output: estimated impact of output of layer on output
                           of network in shape (batch_size, number of channels,
                           height, width)
    :param Numlike activation: estimated activation of input
    :param input_shape: shape of layer input in format
                        (number of batches, number of channels, height, width)
    :type input_shape: tuple of 4 integers
    :param int local_range: Local channel range. Should be odd, otherwise it
                            will be incremented.
    :param float k: Additive constant.
    :param float alpha: The scaling parameter.
    :param float beta: The exponent.
    :returns: Estimated impact of input on output of network
    :rtype: Numlike
    """
    assert_numlike(activation)
    assert_numlike(output)
    res = output.op_d_norm(activation, input_shape, local_range, k, alpha,
                           beta)
    return res


