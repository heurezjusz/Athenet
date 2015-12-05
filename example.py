"""Example usage of Athena network library."""

import theano.tensor as T

from athena_theano.network import Network, FullyConnectedLayer, ReLU, Softmax
from athena_theano.data_loader import load_mnist_data


def output_function(y):
    return T.argmax(y, axis=1)

layers = [
    FullyConnectedLayer(n_in=28*28, n_out=500),
    ReLU(),
    FullyConnectedLayer(n_in=500, n_out=10),
    Softmax(),
]
x = T.matrix('x')
y = T.ivector('y')

network = Network(layers, x, y, output_function)

datasets = load_mnist_data('data/mnist.pkl.gz')
network.train(datasets)
