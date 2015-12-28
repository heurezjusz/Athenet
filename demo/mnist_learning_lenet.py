"""Training LeNet on MNIST data."""

from athena import Network
from athena.layers import ReLU, Softmax, MaxPool, FullyConnectedLayer, \
    ConvolutionalLayer
from athena.utils import load_mnist_data


network = Network([
    ConvolutionalLayer(image_shape=(28, 28, 1), filter_shape=(5, 5, 20)),
    ReLU(),
    MaxPool(poolsize=(2, 2)),
    ConvolutionalLayer(filter_shape=(5, 5, 50)),
    ReLU(),
    MaxPool(poolsize=(2, 2)),
    FullyConnectedLayer(n_out=500),
    ReLU(),
    FullyConnectedLayer(n_out=10),
    Softmax(),
])

network.datasets = load_mnist_data()
network.train(learning_rate=0.1, n_epochs=10, batch_size=300)
