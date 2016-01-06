import unittest
from src.sparsifying.sparsify_smallest import sparsify_smallest_on_layers, sparsify_smallest_on_network
import numpy as np
from mocks.mock_network import NetworkMock, LayerMock
from nose.tools import assert_equal

class SparsifySmallestTest(unittest.TestCase):
    
    def test_no_change_on_network(self):
        layers = [LayerMock(
                        weights=np.random.uniform(low=-15, high=30, size=30),
                        biases=np.random.uniform(low=-15, high=30, size=30)) 
                  for x in xrange(5)]
        network = NetworkMock(weighted_layers=layers)
        sparsify_smallest_on_network(network, 0)
        for layer_new, layer_old in zip(network.weighted_layers, layers):
            assert_equal(layer_new, layer_old)
        
    def test_no_change_on_layers(self):
        layers = [LayerMock(
                        weights=np.random.uniform(low=-15, high=30, size=30),
                        biases=np.random.uniform(low=-15, high=30, size=30)) 
                  for x in xrange(5)]
        network = NetworkMock(weighted_layers=layers)
        sparsify_smallest_on_layers(network, 0)
        for layer_new, layer_old in zip(network.weighted_layers, layers):
            assert_equal(layer_new, layer_old)
    
    def test_zero_for_alls_on_network(self):
        layers = [LayerMock(
                        weights=np.random.uniform(low=-15, high=30, size=30),
                        biases=np.random.uniform(low=-15, high=30, size=30)) 
                  for x in xrange(5)]
        network = NetworkMock(weighted_layers=layers)
        sparsify_smallest_on_network(network, 100)
        for layer in network.weighted_layers:
            for weight in layer.W.flat:
                assert_equal(weight, 0)
    
    def test_zero_for_alls_on_layers(self):
        layers = [LayerMock(
                        weights=np.random.uniform(low=-15, high=30, size=30),
                        biases=np.random.uniform(low=-15, high=30, size=30)) 
                  for x in xrange(5)]
        network = NetworkMock(weighted_layers=layers)
        sparsify_smallest_on_layers(network, 100)
        for layer in network.weighted_layers:
            for weight in layer.W.flat:
                assert_equal(weight, 0)