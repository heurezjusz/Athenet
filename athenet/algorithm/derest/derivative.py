"""For any neuron, we calculate its range of derivative of output with respect
to this neuron. We do it from the end to the beginning of the network. """

import numpy as np
import theano
import theano.tensor as T
from athenet.algorithm.numlike import Numlike, assert_numlike

# TODO: All functions below will be implemented.


def conv(layer_output):
    """Returns estimated impact of convolutional layer on output of network.
    """
    assert_numlike(layer_output)
    pass


def dropout(layer_output):
    """Returns estimated impact of dropout layer on output of network."""
    assert_numlike(layer_output)
    pass


def fully_connected(layer_output):
    """Returns estimated impact of fully connected layer on output of network.
    """
    assert_numlike(layer_output)
    pass


def norm(layer_output):
    """Returns estimated impact of LRN layer on output of network."""
    assert_numlike(layer_output)
    pass


def avg_pool(layer_output):
    """Returns estimated impact of avg pool layer on output of network."""
    assert_numlike(layer_output)
    pass


def max_pool(layer_output):
    """Returns estimated impact of max pool layer on output of network."""
    assert_numlike(layer_output)
    pass


def softmax(layer_output):
    """Returns estimated impact of softmax layer on output of network."""
    assert_numlike(layer_output)
    pass
