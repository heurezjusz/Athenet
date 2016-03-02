"""For any neuron, we calculate its range of activation for inputs of neural
networks in given range. For simplicity, we assume it is [0, 255]. We do it
from the beginning to the end of the network."""

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal import downsample


def conv(self):
    # Might require talk
    pass

def dropout(self):
    pass

def fully_connected(self):
    # Might require talk
    pass

def norm(self):
    pass

def pool(layer_input, poolsize, stride):
    # TODO: To be tested. 
    res_lower = downsample.max_pool_2d(
            input=layer_input.lower,
            ds=poolsize,
            ignore_border=True,
            tr=self.stride
    )
    res_upper = downsample.max_pool_2d(
            input=layer_input.upper,
            ds=poolsize,
            ignore_border=True,
            tr=self.stride
    )
    return Interval(res_lower, res_upper)

def softmax(self):
    pass
