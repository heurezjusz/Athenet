"""Training LeNet on MNIST data."""

from athena import Network
from athena.layers import ReLU, Softmax, MaxPool, FullyConnectedLayer, \
    ConvolutionalLayer
from athena.utils import load_mnist_data


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

network.datasets = load_mnist_data()
network.train(learning_rate=0.1, n_epochs=10, batch_size=300)
