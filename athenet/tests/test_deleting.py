import numpy
import unittest
from copy import deepcopy
from nose.tools import assert_true, assert_false

from athenet.algorithm.deleting import delete_weights_by_layer_fractions,\
    delete_weights_by_global_fraction
from athenet.tests.utils import get_fraction_of_zeros_in_layer,\
    get_fraction_of_zeros_in_network, get_random_layer_mock,\
    get_random_network_mock


class DeletingByFractionTest(unittest.TestCase):
    shapes = [100, (1, 3, 4), (55, 33), (5, 5, 5, 5)]

    def get_random_indicators(self, shape):
        return numpy.random.uniform(low=0, high=1, size=shape)

    def test_layer_no_deleting(self):
        for shape in self.shapes:
            layer = get_random_layer_mock(shape)
            indicators = [self.get_random_indicators(shape)]

            layer_after_deleting = deepcopy(layer)
            delete_weights_by_layer_fractions([layer_after_deleting],
                                              0, indicators)

            assert_true(numpy.equal(layer.W, layer_after_deleting.W).all())

    def test_layer_all_deleting(self):
        for shape in self.shapes:
            layer = get_random_layer_mock(shape)
            indicators = [self.get_random_indicators(shape)]

            layer_after_deleting = deepcopy(layer)
            delete_weights_by_layer_fractions([layer_after_deleting],
                                              1, indicators)

            assert_false(layer_after_deleting.W.any())

    def test_network_by_layers_fraction_deleting(self):
        random_fractions = numpy.random.uniform(low=0, high=1, size=5)
        for fraction in random_fractions:
            for number_of_layers in [1, 3, 6, 10, 20]:
                for shape in self.shapes:
                    network = get_random_network_mock(
                        number_of_layers=number_of_layers,
                        shape_of_layer=shape
                    )
                    indicators = [self.get_random_indicators(shape)
                                  for x in xrange(number_of_layers)]

                    network_after_deleting = deepcopy(network)
                    delete_weights_by_layer_fractions(
                        network_after_deleting.weighted_layers,
                        fraction, indicators
                    )

                    for layer in network_after_deleting.weighted_layers:
                        new_zeros = get_fraction_of_zeros_in_layer(layer)
                        difference = abs(new_zeros - fraction)
                        assert_true(difference <= 1. / layer.W.size)

    def test_network_by_global_fraction_deleting(self):
        random_fractions = numpy.random.uniform(low=0, high=1, size=5)
        for fraction in random_fractions:
            for number_of_layers in [1, 3, 6, 10, 20]:
                for shape in self.shapes:
                    network = get_random_network_mock(
                        number_of_layers=number_of_layers,
                        shape_of_layer=shape
                    )
                    indicators = [self.get_random_indicators(shape)
                                  for x in xrange(number_of_layers)]

                    network_after_deleting = deepcopy(network)
                    delete_weights_by_global_fraction(
                        network_after_deleting.weighted_layers,
                        fraction, indicators
                    )

                    new_zeros = get_fraction_of_zeros_in_network(
                        network=network_after_deleting
                    )
                    difference = abs(new_zeros - fraction)
                    network_size = sum(
                        [layer.W.size for layer
                         in network_after_deleting.weighted_layers]
                    )
                    assert_true(difference <= 1. / network_size)


if __name__ == '__main__':
    unittest.main(verbosity=2, catchbreak=True)
