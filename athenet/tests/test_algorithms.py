import theano
import theano.tensor as T
import numpy as np
from unittest import TestCase, main
from copy import deepcopy

from athenet import Network
from athenet.layers import ConvolutionalLayer, Softmax, FullyConnectedLayer, \
    ReLU, MaxPool
from athenet.utils import DataLoader

from athenet.algorithm import simple_neuron_deleter, middlelayer_neuron_deleter

def get_simple_network():
    result = Network([
        ConvolutionalLayer((5, 5, 30), image_shape=(30, 30, 1)),
        ReLU(),
        MaxPool((2, 2)),
        ConvolutionalLayer((5, 5, 20)),
        ReLU(),
        MaxPool((2, 2)),
        FullyConnectedLayer(100),
        ReLU(),
        FullyConnectedLayer(75),
        ReLU(),
        FullyConnectedLayer(50),
        ReLU(),
        FullyConnectedLayer(10)
    ])
    return result


class TestSender(TestCase):
    def test_algorithm(self):
        net = get_simple_network()
        for layer in net.weighted_layers:
            layer.W = np.ones(layer.W.shape)
            print layer.W.shape
        W = net.weighted_layers[-1].W
        W[0][1] = 200
        net.weighted_layers[-1].W = W
        middlelayer_neuron_deleter(net, (1., 1.))
        for layer in net.weighted_layers:
            print layer.W

if __name__ == '__main__':
    main(verbosity=2, catchbreak=True)
