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
    # TODO: Check if the kernel should be flipped!
    # TODO: Norm _get_output typo, 31
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
    h, w, n_in = input_shape
    fh, fw, n_out = filter_shp
    g_in = n_in / n_groups
    g_out = n_out / n_groups
    pad_h, pad_w = padding
    stride_h, stride_w = stride
    group_image_shape = (g_in, h + 2 * pad_h, w + 2 * pad_w)
    group_filter_shape = (g_out, g_in, fh, fw)
    flipped_weights = weights[:, :, ::-1, ::-1]
    output_h = (h + 2 * pad_h - fh) / stride_h + 1
    output_w = (w + 2 * pad_w - fw) / stride_w + 1
    output_shp = (n_out, output_h, output_w)
    input_type = type(layer_input)
    
    for at_g in range(0, n_groups):
        at_in_from = at_g * g_in
        at_in_to = at_in_from + g_in
        at_out_from = at_g * g_out
        at_out_to = at_out_from + g_out
        for at_h in range(0, group_image_shape[1], stride_h):
            row = []
            for at_w in range(0, group_image_shape[2], stride_w):
                in_slice = layer_input[at_in_from:at_in_to,
                                       at_h:(at_h + fh),
                                       at_w:(at_w + fw)]
                weights_slice = weights[at_out_from:at_out_to, :, :, :]
                conv_sum = in_slice * weight_slice
                conv_sum = conv_sum.sum(axis=1).sum(axis=1).sum(axis=1)
                row += conv_sum

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
