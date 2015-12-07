"""Example usage of Athena network library."""

import theano.tensor as T

from athena_theano.network import (Network, ReLU, Softmax, Activation,
                                   MaxPool, FullyConnectedLayer,
                                   ConvolutionalLayer)
from athena_theano.data_loader import load_mnist_data


network = Network([
    ConvolutionalLayer(image_size=(28, 28), filter_shape=(20, 1, 5, 5)),
    MaxPool(poolsize=(2, 2)),
    ReLU(),
    ConvolutionalLayer(image_size=(12, 12), filter_shape=(50, 20, 5, 5)),
    MaxPool(poolsize=(2, 2)),
    ReLU(),
    FullyConnectedLayer(n_in=50*4*4, n_out=500),
    ReLU(),
    FullyConnectedLayer(n_in=500, n_out=10),
    Softmax(),
])

datasets = load_mnist_data('data/mnist.pkl.gz')
network.train(datasets=datasets)
