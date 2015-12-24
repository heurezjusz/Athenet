"""Training LeNet on MNIST data."""

import os
import numpy as np

from athena import Network
from athena.layers import ReLU, Softmax, MaxPool, FullyConnectedLayer, \
    ConvolutionalLayer
from athena.utils.data_loader import load_mnist_data, download_mnist_data


network = Network([
    ConvolutionalLayer(image_size=(28, 28), filter_shape=(20, 1, 5, 5)),
    ReLU(),
    MaxPool(poolsize=(2, 2)),
    ConvolutionalLayer(image_size=(12, 12), filter_shape=(50, 20, 5, 5)),
    ReLU(),
    MaxPool(poolsize=(2, 2)),
    FullyConnectedLayer(n_in=50*4*4, n_out=500),
    ReLU(),
    FullyConnectedLayer(n_in=500, n_out=10),
    Softmax(),
])

mnist_filename = '../bin/mnist.pkl.gz'
if not os.path.isfile(mnist_filename):
    download_mnist_data(mnist_filename)

network.datasets = load_mnist_data(mnist_filename)
network.train(learning_rate=0.1, n_epochs=1, batch_size=300)
