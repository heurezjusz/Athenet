"""Example usage of Athena network library."""

import os
import numpy as np

from athena.network import Network
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

mnist_filename = 'mnist.pkl.gz'
if not os.path.isfile(mnist_filename):
    download_mnist_data(mnist_filename)

# learning example
network.datasets = load_mnist_data(mnist_filename)
network.train(learning_rate=0.1, n_epochs=2, batch_size=300)

# weights modifying example
W = network.weighted_layers[0].W  # get copy of the weights' values
W += np.random.uniform(low=0.0, high=0.1, size=W.shape)  # random disturbance
network.weighted_layers[0].W = W  # set the new weights
print 'Accuracy on the test data after disturbance: {:.2f}%'.format(
    100 * network.test_accuracy())
