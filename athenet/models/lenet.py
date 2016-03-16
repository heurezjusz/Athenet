"""Functions returning out-of-the-box LeNet, instance of Network"""

from athenet import Network
from athenet.layers import ReLU, Softmax, MaxPool, FullyConnectedLayer, \
    ConvolutionalLayer
from athenet.utils import load_data, get_bin_path

LENET_FILENAME = 'lenet_weights.pkl.gz'
LENET_URL = 'http://students.mimuw.edu.pl/~wg346897/hosting/athenet/' \
    'lenet_weights.pkl.gz'


def lenet(trained=True, weights_filename=LENET_FILENAME,
          weights_url=LENET_URL):
    """Create and return instance of LeNet network.

    :trained: If True, trained weights will be loaded from file.
    :weights_filename: Name of a file with LeNet weights. Will be used if
                       ``trained`` argument is set to True.
    :weights_url: Url from which to download file with weights.
    :return: LeNet network.
    """
    if trained:
        weights = load_data(get_bin_path(weights_filename), weights_url)
        if weights is None:
            raise Exception("cannot load LeNet weights")

    lenet = Network([
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
    if trained:
        lenet.set_params(weights)
    return lenet
