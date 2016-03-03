"""Testing athenet.sparsifying.derest.activation functions.
"""

import unittest
from nose.tools import assert_true, assert_is, assert_equal
import numpy as np
from numpy.testing import assert_array_equal as are, \
    assert_array_almost_equal as arae
from theano import function
import theano.tensor as T
from athenet.sparsifying.utils.interval import Interval as I
from athenet.sparsifying.derest.activation import *

A = np.array

class FullyConnectedActivationTest(unittest.TestCase):

    def prepare(self):
        self.v = np.arange(24) + 3.0
        self.at_v = 0
        return self.s, self.v

    def s(self):
        if self.at_v >= len(self.v):
            raise TypeError
        ret = self.v[self.at_v]
        self.at_v += 1
        return ret

    def test1DSimple(self):
        s, v = self.prepare()
        #arae(fully_connected(1, 2, 0), 2)

    def test2DSimpleUsed1DOfWeights(self):
        s, v = self.prepare()
        print s(), s(), s(), s()
        inp, w, b = A([[s()], [s()]]), A([[s()], s()]), 1.0
        #arae(fully_connected(inp, w, b), v[0] * v[2] + v[1] * v[3] + 1.0)

    def test2DSimpleUSed2DOfWeights(self):
        s, v = self.prepare()
        #v = self.v
        i = A([v[0], v[1]])

    def test2DSimple(self):
        s, v = self.prepare()
        pass
        #v = self.v

    def test3D(self):
        s, v = self.prepare()
        v = self.v

    def test3DUsingIntervals(self):
        s, v = self.prepare()
        v = self.v


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

if __name__ == '__main__':
    unittest.main(verbosity=2, catchbreak=True)
