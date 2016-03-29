import unittest
import numpy
from copy import deepcopy
from nose.tools import assert_false, assert_true

from athenet.algorithm.sharpen_filters import sharpen_filters_in_layer
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

    def test_no_change(self):
        layer = self.get_conv_layer()
        changed_layer = deepcopy(layer)
        sharpen_filters_in_layer(changed_layer, (1, 0, (5, 75, 75)))
        assert_true(numpy.array_equal(layer.W, changed_layer.W))

    def test_all_zeros(self):
        layer = self.get_conv_layer()
        sharpen_filters_in_layer(layer, (0, 1, (5, 75, 75)))
        assert_false(layer.W.any())

    def test_flat_filter(self):
        layer = LayerMock(
            weights=numpy.full(self.shape, 0.4),
            biases=numpy.random.uniform(low=-1, high=1, size=self.shape)
        )

        for min_noise, max_value in [(0.1, 0.2), (0.1, 0.6), (0, 0.2)]:
            changed_layer = deepcopy(layer)
            sharpen_filters_in_layer(changed_layer,
                                     (min_noise, max_value, (5, 75, 75)))
            assert_true(numpy.array_equal(layer.W, changed_layer.W))

        changed_layer = deepcopy(layer)
        sharpen_filters_in_layer(changed_layer, (0, 0.6, (5, 75, 75)))
        assert_false(changed_layer.W.any())

    def test_small_values_in_filters(self):
        layer = LayerMock(
            weights=numpy.random.uniform(low=-0.2, high=0.2, size=self.shape),
            biases=numpy.random.uniform(low=-1, high=1, size=self.shape)
        )

        changed_layer = deepcopy(layer)
        sharpen_filters_in_layer(changed_layer, (0.1, 0.1, (5, 75, 75)))
        assert_true(changed_layer.W.any())

        changed_layer = deepcopy(layer)
        sharpen_filters_in_layer(changed_layer, (0, 0.3, (5, 75, 75)))
        assert_false(changed_layer.W.any())
