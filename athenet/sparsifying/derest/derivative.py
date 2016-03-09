"""For any neuron, we calculate its range of derivative of output with respect
to this neuron. We do it from the end to the beginning of the network. """

import numpy as np
import theano
import theano.tensor as T
    
# TODO: All functions below will be implemented.

def conv():
    """Returns estimated impact of convolutional layer on output of network."""
    pass

def dropout():
    """Returns estimated impact of dropout layer on output of network."""
    pass

def fully_connected():
    """Returns estimated impact of fully connected layer on output of network."""
    pass

def norm():
    """Returns estimated impact of LRN layer on output of network."""
    pass

def pool():
    """Returns estimated impact of max pool layer on output of network."""
    pass

def softmax():
    """Returns estimated impact of softmax layer on output of network."""
    pass
