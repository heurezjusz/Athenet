import unittest
import numpy
from nose.tools import assert_equal, assert_false

from athenet.algorithm.sharpen_filters import get_filters_indicators
from athenet.tests.utils import get_random_layer_mock, get_random_network_mock


class SharpenFiltersIndicatorsTest(unittest.TestCase):
    bilateral_filter_args = (5, 75, 75)

    def test_sharpen_filters_indicators_shape(self):
        for number_of_layers in [1, 3, 5, 7, 10, 20]:
            for layers_shape in [(1, 1, 5, 5), (3, 5, 5, 30), (6, 1, 10, 10)]:
                network = get_random_network_mock(
                    number_of_layers=number_of_layers,
                    shape_of_layer=layers_shape
                )
                indicators = get_filters_indicators(
                    layers=network.weighted_layers,
                    bilateral_filter_args=self.bilateral_filter_args
                )

                assert_equal(number_of_layers, len(indicators))
                for layer in xrange(number_of_layers):
                    assert_equal(layers_shape, indicators[layer].shape)

    def test_sharpen_filters_indicators_in_range(self):
        layer = get_random_layer_mock((6, 2, 10, 10))
        indicators = get_filters_indicators(
            layers=[layer],
            bilateral_filter_args=self.bilateral_filter_args
        )

        assert_false(numpy.any(indicators > 1))
        assert_false(numpy.any(indicators < 0))

    def test_sharpen_filters_indicators_on_fully_connected(self):
        layer = get_random_layer_mock(100)
        indicators = get_filters_indicators(
            layers=[layer],
            bilateral_filter_args=self.bilateral_filter_args
        )
        assert_equal(len(indicators), 0)
