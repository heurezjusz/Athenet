"""Example usage of Athena network library."""

from __future__ import print_function
import os
import urllib

from athena_theano.network import (Network, ReLU, Softmax, MaxPool,
                                   FullyConnectedLayer, ConvolutionalLayer)
from athena_theano.data_loader import load_mnist_data


def download_mnist_data(filename):
    """Download MNIST data.

    filename: Name of the MNIST data file to be created
    """
    print('Downloading MNIST data...', end=' ')
    mnist_origin = ('http://www.iro.umontreal.ca/~lisa/deep/data/mnist/'
                    'mnist.pkl.gz')
    urllib.urlretrieve(mnist_origin, filename)
    print('Done.')

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

mnist_filename = 'mnist.pkl.gz'
if not os.path.isfile(mnist_filename):
    download_mnist_data(mnist_filename)
datasets = load_mnist_data(mnist_filename)
network.set_training_data(datasets)

network.train(learning_rate=0.1, n_epochs=10, batch_size=300)
