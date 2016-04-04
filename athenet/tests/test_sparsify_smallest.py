import unittest
from random import random

import numpy as np
from nose.tools import assert_equal, assert_true

from athenet.algorithm.sparsify_smallest import sparsify_smallest_on_layers,\
    sparsify_smallest_on_network, get_smallest_indicators
from athenet.tests.mock_network import NetworkMock, LayerMock
from athenet.tests.utils import get_fraction_of_zeros_in_layer,\
    get_fraction_of_zeros_in_network, get_random_layer_mock,\
    get_random_network_mock


class SparsifySmallestIndicatorsTest(unittest.TestCase):

    def test_smallest_indicator_size(self):
        for number_of_layers in [1, 3, 5, 7, 10, 20]:
            for size_of_layer in [1, 10, 100, 500]:
                network = get_random_network_mock(
                    number_of_layers=number_of_layers,
                    shape_of_layer=size_of_layer)
                indicators = get_smallest_indicators(network.weighted_layers)

                assert_equal(number_of_layers, len(indicators))
                for layer in xrange(number_of_layers):
                    assert_equal(size_of_layer, len(indicators[layer]))

    def test_smallest_indicators_equal(self):
        weight = random()
        network = NetworkMock(
            weighted_layers=[LayerMock(
                weights=np.full((100), weight),
                biases=np.random.uniform(low=-1, high=1, size=100)
            ) for x in xrange(7)]
        )
        indicators = get_smallest_indicators(network.weighted_layers)
        assert_equal(np.amin(indicators), np.amax(indicators))

    def test_smallest_indicators_min_max(self):
        layer = get_random_layer_mock()
        indicators = get_smallest_indicators([layer])

        assert_equal(np.argmax(abs(layer.W)), np.argmin(indicators[0]))
        assert_equal(np.argmin(abs(layer.W)), np.argmax(indicators[0]))


class SparsifySmallestTest(unittest.TestCase):
    epsilon = 10 ** (-5)

    def setUp(self):
        self.network = get_random_network_mock()

    def test_no_change_on_network(self):
        layers_weights = [np.copy(layer.W)
                          for layer in self.network.weighted_layers]
        sparsify_smallest_on_network(self.network, 0)
        for layer_new, layer_old \
                in zip(self.network.weighted_layers, layers_weights):
            assert_true(np.array_equal(layer_new.W, layer_old))

    def test_no_change_on_layers(self):
        layers_weights = [np.copy(layer.W)
                          for layer in self.network.weighted_layers]
        sparsify_smallest_on_layers(self.network, 0)
        for layer, old_weights \
                in zip(self.network.weighted_layers, layers_weights):
            assert_true(np.array_equal(layer.W, old_weights))

    def test_zero_for_all_on_network(self):
        sparsify_smallest_on_network(self.network, 1)
        assert_equal(get_fraction_of_zeros_in_network(self.network), 1)

    def test_zero_for_all_on_layers(self):
        sparsify_smallest_on_layers(self.network, 1)
        for layer in self.network.weighted_layers:
            assert_equal(get_fraction_of_zeros_in_layer(layer), 1)

    def test_fraction_of_zeros_on_network(self):
        for fraction in [0.1, 0.3, 0.5, 0.6, 0.9]:
            sparsify_smallest_on_network(self.network, fraction)
            zeros_fraction = get_fraction_of_zeros_in_network(self.network)
            difference = abs(zeros_fraction - fraction)
            network_size = sum([layer.W.size
                                for layer in self.network.weighted_layers])
            assert_true(difference <= 1. / network_size)

    def test_fraction_of_zeros_on_layers(self):
        for fraction in [0.1, 0.3, 0.5, 0.6, 0.9]:
            sparsify_smallest_on_layers(self.network, fraction)
            for layer in self.network.weighted_layers:
                zeros_fraction = get_fraction_of_zeros_in_layer(layer)
                difference = abs(zeros_fraction - fraction)
                assert_true(difference <= 1. / layer.W.size)

if __name__ == '__main__':
    unittest.main(verbosity=2, catchbreak=True)
