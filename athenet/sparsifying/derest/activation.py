"""For any neuron, we calculate its range of activation for inputs of neural
networks in given range. For simplicity, we assume it is [0, 255]. We do it
from the beginning to the end of the network. Functions at first try to use
"""

import numpy as np
import theano
import theano.tensor as T
from athenet.sparsifying.derest.utils import *

# TODO: All functions below will be implemented.

def conv(layer_input, input_shp, weights, weights_shape, stride=(1, 1),
        padding=(0, 0), n_groups=1):
    """Returns estimated activation of convolutional layer.

    :layer_input: Input tensor
    :input_shp: Shape of input in the format
                (image height, image width, number of input channels)
    :weights: weights tensor
    :weights_shp: Weights shape in the format
                  (filter height, filter width, number of output channels)
    :stride: Pair representing interval at which to apply the filters.
    :padding: Pair representing number of zero-valued pixels to add on each
              side of the input.
    :n_groups: Number of groups input and output channels will be split
               into. Two channels are connected only if they belong to the
               same group.
    """
    assert_numlike(layer_input)
    try:
        h, w, n_in = input_shape
        fh, fw, n_out = weights_shape
        group_in = n_in / n_groups
        group_out = n_out / n_groups
        pad_h, pad_w = padding
        #group_image


    #TODO
    h, w = self.image_shape[0:2]
    pad_h, pad_w = self.padding
    group_image_shape = (self.batch_size, n_group_channels,
                         h + 2*pad_h, w + 2*pad_w)
    h, w = self.filter_shape[0:2]
    group_filter_shape = (n_group_filters, n_group_channels, h, w)

    conv_outputs = [theano.tensor.nnet.conv.conv2d(
        input=self.input[:, i*n_group_channels:(i+1)*n_group_channels,
                         :, :],
        filters=self.W_shared[i*n_group_filters:(i+1)*n_group_filters,
                              :, :, :],
        filter_shape=group_filter_shape,
        image_shape=group_image_shape,
        subsample=self.stride
    ) for i in xrange(self.n_groups)]


    except:

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
