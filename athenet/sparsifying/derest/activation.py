"""For any neuron, we calculate its range of activation for inputs of neural
networks in given range. For simplicity, we assume it is [0, 255]. We do it
from the beginning to the end of the network. Functions at first try to use
"""

import numpy as np
import theano
import theano.tensor as T
from athenet.sparsifying.derest.utils import *

# TODO: All functions below will be implemented.

def conv(layer_input, input_shp, weights, filter_shp, biases, stride=(1, 1),
        padding=(0, 0), n_groups=1):
    # TODO: unit tests
    """Returns estimated activation of convolutional layer.

    :param layer_input: Input Numlike
    :param input_shp: Shape of input in the format
                (number of input channels, image height, image width)
    :param weights: Weights tensor in format (number of output channels,
                                              number of input channels,
                                              filter height, filter width)
    :param filter_shp: Filter shape in the format (number of output channels,
                                                   filter height, filter width)
    :param biases: Biases in convolution
    :param stride: Pair representing interval at which to apply the filters.
    :param padding: Pair representing number of zero-valued pixels to add on each
                    side of the input.
    :param n_groups: Number of groups input and output channels will be split
                     into. Two channels are connected only if they belong to the
                     same group.
    :type layer_input: Numlike or numpy.ndarray or theano tensor
    :type input_shp: integer tuple
    :type weights: numpy.ndarray or theano tensor
    :type filter_shp: integer tuple
    :type biases: 1D np.array or theano.tensor
    :type stride: integer pair
    :type padding: integer pair
    :type n_groups: integer
    :rtype: Numlike
    """
    assert_numlike(layer_input)
    # h, w, n_in - image height, image width, number of input channels
    h, w, n_in = input_shp
    # fw, fh, n_out - filter height, filter width, number of output channels
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
                input_slice = padded_input[at_in_from:at_in_to,
                                       at_h:(at_h + fh),
                                       at_w:(at_w + fw)]
                # weights slice that impacts on (at_out_h, at_out_w) in output
                weights_slice = flipped_weights[at_out_from:at_out_to, :, :, :]
                conv_sum = input_slice * weight_slice
                conv_sum = conv_sum.sum(axis=(1, 2, 3), keepdims=True)
                result[at_out_from:at_out_to, at_out_h, at_out_w] = conv_sum
    result = result + bias
    return result

def dropout(layer_input, p_dropout):
    """Returns estimated activation of dropout layer.

    :param p_drouput: probability of dropping in dropout
    :type p_dropout: float
    :rtype: Numlike
    """
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

def pool(layer_input, poolsize, stride=None, mode="max"):
    """Returns estimated activation of max pool layer."""
    # TODO: mode 'avg'
    assert_numlike(input_layer)
    if stride is None:
        stride = poolsize
    if mode not in ['max', 'avg']:
        raise ValueError("mode not in ['max', 'avg']")
    is_max = mode == 'max'
    # h, w, n_in - image height, image width, number of input channels
    h, w, n_in = input_shape
    stride_h, stride_w = stride
    pool_h, pool_w = poolsize
    output_h = (h - pool_h) / stride_h + 1
    output_w = (w - pool_w) / stride_w + 1
    output_shp = (n_out, output_h, output_w)
    result = input_type.from_shape(output_shp)
    for at_h in range(0, h, stride_h):
        at_out_h = at_h / stride_h
        at_h_from = at_h
        at_h_to = at_h + stride_h
        for at_w in range(0, w, stride_w):
            at_w_from = at_w
            at_w_to = at_w + stride_w
            at_out_w = at_w / stride_w
            input_slice = layer_input[:, at_h_from:at_h_to, at_w_from:at_w_to]
            if is_max:
                pool_res = input_slice.amax(axis=(1, 2), keepdims=True)
            else:
                # TODO
                pass
            result[:, at_out_h, at_out_w] = pool_res
    return result

def softmax(layer_input):
    """Returns estimated activation of softmax layer."""
    assert_numlike(input_layer)
#    try:
#    except:
