"""For any neuron, we calculate its range of activation for inputs of neural
networks in given range. For simplicity, we assume it is [0, 255]. We do it
from the beginning to the end of the network. Functions at first try to use
"""

import numpy as np
import theano
import theano.tensor as T
from athenet.sparsifying.derest.utils import *

# TODO: All functions below will be implemented.

def conv(layer_input, input_shp, weights, filter_shp, stride=(1, 1),
        padding=(0, 0), n_groups=1):
    # TODO: unit tests
    """Returns estimated activation of convolutional layer.

    :param layer_input: input Interval
    :param input_shp: Shape of input in the format
                (number of input channels, image height, image width)
    :param weights: weights tensor in format (number of output channels,
                                              number of input channels,
                                              filter height, filter width)
    :param filter_shp: Filter shape in the format (number of output channels,
                                                   filter height, filter width)
    :param stride: Pair representing interval at which to apply the filters.
    :param padding: Pair representing number of zero-valued pixels to add on each
                    side of the input.
    :param n_groups: Number of groups input and output channels will be split
                     into. Two channels are connected only if they belong to the
                     same group.
    :type layer_input: Interval or numpy.ndarray or theano tensor
    :type input_shp: integer tuple
    :type weights: numpy.ndarray or theano tensor
    :type filter_shp: integer tuple
    :type stride: integer pair
    :type padding: integer pair
    :type n_groups: integer
    :rtype: Interval
    """
    assert_numlike(layer_input)
    # h, w, n_in - image height, image width, number of input channels
    h, w, n_in = input_shape
    # fh, fw, n_out - filter height, filter width, number of output channels
    fh, fw, n_out = filter_shp
    # g_in - number of input channels per group
    g_in = n_in / n_groups
    # g_out - number of output channels per group
    g_out = n_out / n_groups
    pad_h, pad_w = padding
    stride_h, stride_w = stride
    # see: flipping kernel
    flipped_weights = weights[:, :, ::-1, ::-1]
    input_type = type(layer_input)
    padded_input_shape = (h + 2 * pad_h, w + 2 * pad_w, n_in)
    padded_input = input_type.from_shape(padded_input_shape)
    padded_input[pad_h:(pad_h + h), pad_w:(pad_w + w), n_in] = \
            layer_input
    # setting new h, w, n_in for padded input, you can forget about padding
    h, w, n_in = padded_input_shape
    output_h = (h - fh) / stride_h + 1
    output_w = (w - fw) / stride_w + 1
    output_shp = (n_out, output_h, output_w)
    result = input_type.from_shape(output_shp)
    for at_g in range(0, n_groups):
        # beginning and end of at_g'th group of input channel in input
        at_in_from = at_g * g_in
        at_in_to = at_in_from + g_in
        # beginning and end of at_g'th group of output channel in weights
        at_out_from = at_g * g_out
        at_out_to = at_out_from + g_out
        for at_h in range(0, h, stride_h):
            # at_out_h - height of output corresponding to filter at
            # position at_h
            at_out_h = at_h / stride_h
            for at_w in range(0, w, stride_w):
                # at_out_w - height of output corresponding to filter at
                # position at_w
                at_out_w = at_w / stride_w
                # input slice that impacts on (at_out_h, at_out_w) in output
                input_slice = layer_input[at_in_from:at_in_to,
                                       at_h:(at_h + fh),
                                       at_w:(at_w + fw)]
                # weights slice that impacts on (at_out_h, at_out_w) in output
                weights_slice = flipped_weights[at_out_from:at_out_to, :, :, :]
                conv_sum = input_slice * weight_slice
                conv_sum = conv_sum.sum(axis=1).sum(axis=1).sum(axis=1)
                conv_sum = conv_sum.reshape((g_out, 1, 1))
                result[at_out_from:at_out_to, at_out_h, at_out_w] = conv_sum
    return result

def dropout(layer_input, p_dropout):
    """Returns estimated activation of dropout layer."""
    assert_numlike(layer_input)
    try:
        return layer_input.op_dropout(p_dropout)
    except:
        return layer_input * (1.0 - p_dropout)

def fully_connected(layer_input, weights, biases):
    """Returns estimated activation of fully connected layer."""
    assert_numlike(layer_input)
    flat_input = layer_input.flatten()
    try:
        return flat_input.dot(weights) + biases
    except:
        return (flat_input * weights.T).sum(1) + biases

def norm(input_layer, local_range=5, k=1, alpha=0.0002, beta=0.75):
    """Returns estimated activation of LRN layer."""
    assert_numlike(input_layer)
#    try:
#    except:

def avg_pool(layer_input, poolsize, stride=None):
    """Returns estimated activation of avg pool layer."""
    assert_numlike(input_layer)
#    try:
#    except:

def max_pool(layer_input, poolsize, stride=None):
    """Returns estimated activation of max pool layer."""
    assert_numlike(input_layer)
#    try:
#    except:

def softmax(layer_input):
    """Returns estimated activation of softmax layer."""
    assert_numlike(input_layer)
#    try:
#    except:
