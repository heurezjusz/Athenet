"""Testing athenet.sparsifying.derest.activation functions.
"""

import unittest
from nose.tools import assert_true, assert_is, assert_equal
import numpy as np
from numpy.testing import assert_array_equal as are,
                          assert_array_almost_equal as arae
from theano import function
import theano.tensor as T
from athenet.sparsifying.utils.interval import Interval as I
from athenet.sparsifying.derest.activation import *

A = np.array

class FullyConnectedActivationTest(unittest.TestCase):
    #TODO: Setup?
    def setUp(self):
        self.v =  np.arange(24) + 3.0

    def test_1D_simple(self):
        arae(fully_connected(1, 2, 0), 2)

    def test_2d_used_1D_of_weights(self):
        v = self.v
        inp, w, b = 2, A([[v[0]], [v[1]]]), A([[[v[2]]], [v[3]]]), 1.0
        arae(fully_connected(inp, w, b), v[0] * v[2] + v[1] * v[3] + 1.0)

class ConvolutionalActivationTest(unittest.TestCase):
    pass

class MaxPoolActivationTest(unittest.TestCase):
    pass

class SoftmaxActivationTest(unittest.TestCase):
    pass

class LRNActivationTest(unittest.TestCase):
    pass

class DropoutActivationTest(unittest.TestCase):
    pass
