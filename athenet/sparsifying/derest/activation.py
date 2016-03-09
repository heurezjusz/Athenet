"""For any neuron, we calculate its range of activation for inputs of neural
networks in given range. For simplicity, we assume it is [0, 255]. We do it
from the beginning to the end of the network. Functions at first try to use
"""

import numpy as np
import theano
import theano.tensor as T
from athenet.sparsifying.derest.utils import *

# TODO: All functions below will be implemented.

def conv(layer_input, weights, stride=(1, 1), padding=(0, 0), n_groups=1, batch_size=1):
    """Returns estimated activation of convolutional layer."""
    assert_numlike(layer_input)
    #try:
    #except:

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
