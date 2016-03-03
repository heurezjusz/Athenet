"""Testing athenet.sparsifying.derest.activation functions.
"""

import unittest
from nose.tools import assert_true, assert_is, assert_equal
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import theano.tensor as T
from theano import function
from athenet.sparsifying.utils.interval import Interval
from athenet.sparsifying.derest.activation import *

class FullyConnectedDerivativeTest(unittest.TestCase):
    pass

class ConvolutionalDerivativeTest(unittest.TestCase):
    pass

class MaxPoolDerivativeTest(unittest.TestCase):
    pass

class SoftmaxDerivativeTest(unittest.TestCase):
    pass

class LRNDerivativeTest(unittest.TestCase):
    pass

class DropoutDerivativeTest(unittest.TestCase):
    pass
