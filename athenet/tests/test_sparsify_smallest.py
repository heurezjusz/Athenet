import unittest
from athenet.algorithm.sparsify_smallest import sparsify_smallest_on_layers
from athenet.algorithm.sparsify_smallest import sparsify_smallest_on_network
import numpy as np
from mocks.mock_network import NetworkMock, LayerMock
from nose.tools import assert_equal, assert_true


class SparsifySmallestTest(unittest.TestCase):

    def set_network(self, number_of_layers=5, size_of_layer=100):
        layers = [LayerMock(
            weights=np.random.uniform(low=-15, high=30, size=size_of_layer),
            biases=np.random.uniform(low=-15, high=30, size=size_of_layer))
            for x in xrange(number_of_layers)]
        self.network = NetworkMock(weighted_layers=layers)

    def get_percentage_of_zeros_in_layer(self, layer):
        return 100 - 100 * np.count_nonzero(layer.W.flat) / len(layer.W.flat)

    def get_percentage_of_zeros_in_network(self, network):
        number_of_nonzeros = sum((np.count_nonzero(layer.W.flat)
                                  for layer in network.weighted_layers))
        number_of_weights = sum((len(layer.W.flat)
                                 for layer in network.weighted_layers))
        return 100 - 100 * number_of_nonzeros / number_of_weights

    def test_no_change_on_network(self):
        self.set_network()
        layers_weights = [np.copy(layer.W)
                          for layer in self.network.weighted_layers]
        sparsify_smallest_on_network(self.network, 0)
        for layer_new, layer_old \
                in zip(self.network.weighted_layers, layers_weights):
            assert_true(np.array_equal(layer_new.W, layer_old))

    def test_no_change_on_layers(self):
        self.set_network()
        layers_weights = [np.copy(layer.W)
                          for layer in self.network.weighted_layers]
        sparsify_smallest_on_layers(self.network, 0)
        for layer, old_weights \
                in zip(self.network.weighted_layers, layers_weights):
            assert_true(np.array_equal(layer.W, old_weights))

    def test_zero_for_all_on_network(self):
        self.set_network()
        sparsify_smallest_on_network(self.network, 100)
        assert_equal(self.get_percentage_of_zeros_in_network(self.network),
                     100)

    def test_zero_for_all_on_layers(self):
        self.set_network()
        sparsify_smallest_on_layers(self.network, 100)
        for layer in self.network.weighted_layers:
            assert_equal(self.get_percentage_of_zeros_in_layer(layer), 100)

    def test_percentage_of_zeros_on_network(self):
        self.set_network()
        for percentage in [10, 30, 50, 60, 90]:
            sparsify_smallest_on_network(self.network, percentage)
            assert_true(self.get_percentage_of_zeros_in_network(self.network)
                        == percentage)

    def test_percentage_of_zeros_on_layers(self):
        self.set_network()
        for percentage in [10, 30, 50, 60, 90]:
            sparsify_smallest_on_layers(self.network, percentage)
            for layer in self.network.weighted_layers:
                assert_true(self.get_percentage_of_zeros_in_layer(layer)
                            == percentage)

if __name__ == '__main__':
    unittest.main(verbosity=2, catchbreak=True)
