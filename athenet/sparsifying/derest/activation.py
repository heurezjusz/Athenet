"""For any neuron, we calculate its range of activation for inputs of neural
networks in given range. For simplicity, we assume it is [0, 255]. We do it
from the beginning to the end of the network."""

import numpy as np
import theano
import theano.tensor as T

# TODO: All functions below will be implemented.

def conv():
    """Returns estimated activation of convolutional layer."""
    pass

def dropout(layer_input, p_dropout):
    """Returns estimated activation of dropout layer."""
    return layer_input * (1.0 - p_dropout)

def fully_connected():
    """Returns estimated activation of fully connected layer."""
    pass

def norm():
    """Returns estimated activation of LRN layer."""
    pass

def pool(layer_input, poolsize, stride):
    """Returns estimated activation of max pool layer."""
    pass

def softmax():
    """Returns estimated activation of softmax layer."""
    pass
