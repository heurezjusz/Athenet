"""For any neuron, we calculate its range of activation for inputs of neural
networks in given range. For simplicity, we assume it is [0, 255]. We do it
from the beginning to the end of the network."""

import numpy as np
import theano
import theano.tensor as T

# TODO: All functions below will be implemented.

def conv(layer_input, weights, stride=(1, 1), padding=(0, 0), n_groups=1, batch_size=1):
    """Returns estimated activation of convolutional layer."""
    try:
    except:

def dropout(layer_input, p_dropout):
    """Returns estimated activation of dropout layer."""
    try:
        layer_input.op.dropout(p_dropout)
    except:
        return layer_input * (1.0 - p_dropout)

def fully_connected(layer_input, weights, biases):
    """Returns estimated activation of fully connected layer."""
    try:
    except:
    pass

def norm(input_layer, local_range=5, k=1, alpha=0.0002, beta=0.75):
    """Returns estimated activation of LRN layer."""
    try:
    except:
    pass

def pool(layer_input, poolsize, stride=None):
    """Returns estimated activation of max pool layer."""
    try:
    except:
    pass

def softmax(layer_input):
    """Returns estimated activation of softmax layer."""
    try:
    except:
    pass
