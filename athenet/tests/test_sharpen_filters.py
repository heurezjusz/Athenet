import unittest
import numpy
from copy import deepcopy
from nose.tools import assert_false, assert_true

from athenet.algorithm.sharpen_filters import sharpen_filters
from mocks.mock_network import LayerMock


class SharpenFiltersTest(unittest.TestCase):
    shape = (6, 1, 10, 10)

    def get_conv_layer(self, shape=None):
        if shape is None:
            shape = self.shape
        return LayerMock(
            weights=numpy.random.uniform(low=-1, high=1, size=shape),
            biases=numpy.random.uniform(low=-1, high=1, size=shape)
        )
