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


def a_pool(layer_input, input_shp, poolsize, stride=(1, 1), mode="max"):
    """Returns estimated activation of pool layer.

    :param Numlike layer_input: Numlike input in input_shp format
    :param tuple of 3 integers input_shp: input shape in format (n_channels,
                                          height, width)
    :param pair of integers poolsize: pool size in format (height, width)
    :param pair of integers stride: stride of max pool
    :param 'max' or 'avg' mode: specifies whether it is max pool or average
                                pool
    :rtype: Numlike
    """
    assert_numlike(layer_input)
    if mode not in ["max", "avg"]:
        raise ValueError("pool mode should be 'max' or 'avg'")
    is_max = mode == "max"
    # n_in, h, w - number of input channels, image height, image width
    n_in, h, w = input_shp
    n_out = n_in
    # fh, fw - pool height, pool width
    fh, fw = poolsize
    stride_h, stride_w = stride
    output_h = (h - fh) / stride_h + 1
    output_w = (w - fw) / stride_w + 1
    output_shp = (n_out, output_h, output_w)
    result = layer_input.from_shape(output_shp, neutral=True)
    for at_h in xrange(0, h - fh + 1, stride_h):
        # at_out_h - height of output corresponding to pool at position at_h
        at_out_h = at_h / stride_h
        for at_w in xrange(0, w - fw + 1, stride_w):
            # at_out_w - height of output corresponding to pool at
            # position at_w
            at_out_w = at_w / stride_w
            input_slice = layer_input[:, at_h:(at_h + fh), at_w:(at_w + fw)]
            if is_max:
                pool_res = input_slice.amax(axis=(1, 2), keepdims=False)
            else:
                pool_res = input_slice.sum(axis=(1, 2), keepdims=False) \
                    / float(fh * fw)
            result[:, at_out_h, at_out_w] = pool_res
    return result


def a_softmax(layer_input, input_shp):
    """Returns estimated activation of softmax layer.
    :param Numlike layer_input: input
    :param integer input_shp: shape of 1D input
    :rtype: Numlike
    """
    assert_numlike(layer_input)
    try:
        res = layer_input.op_softmax(input_shp)
    except NotImplementedError:
        exponents = layer_input.exp()
        res = exponents / exponents.sum()
    return res


def a_relu(layer_input):
    """Returns estimated activation of relu layer.

    :param Numlike layer_input: input
    :rtype: Numlike
    """
    assert_numlike(layer_input)
    try:
        res = layer_input.op_relu()
    except NotImplementedError:
        res = (layer_input + layer_input.abs()) * 0.5
    return res
