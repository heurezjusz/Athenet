"""GoogLeNet model."""

from athenet import Network
from athenet.layers import ConvolutionalLayer, ReLU, LRN, MaxPool, AvgPool, \
    FullyConnectedLayer, Dropout, Softmax, InceptionLayer
from athenet.utils import load_data, get_bin_path

GOOGLENET_FILENAME = 'googlenet_weights.pkl.gz'


def googlenet(trained=True, weights_filename=GOOGLENET_FILENAME,
              weights_url=None):
    if trained:
        weights = load_data(get_bin_path(weights_filename), weights_url)
        if weights is None:
            raise Exception("cannot load GoogLeNet weights")

    # Normalization parameters
    local_range = 5
    alpha = 0.0001
    beta = 0.75
    k = 1

    googlenet = Network([
        ConvolutionalLayer(image_shape=(224, 224, 3),
                           filter_shape=(7, 7, 64),
                           stride=(2, 2),
                           padding=(3, 3)),
        ReLU(),
        MaxPool(poolsize=(3, 3),
                stride=(2, 2),
                padding=(1, 1)),
        LRN(local_range=local_range,
            alpha=alpha,
            beta=beta,
            k=k),
        ConvolutionalLayer(filter_shape=(1, 1, 64)),
        ReLU(),
        ConvolutionalLayer(filter_shape=(3, 3, 192),
                           padding=(1, 1)),
        ReLU(),
        LRN(local_range=local_range,
            alpha=alpha,
            beta=beta,
            k=k),
        MaxPool(poolsize=(3, 3),
                stride=(2, 2),
                padding=(1, 1)),
        InceptionLayer([64, 96, 128, 16, 32, 32], name='inception 3a'),
        InceptionLayer([128, 128, 192, 32, 96, 64], name='inception 3b'),
        MaxPool(poolsize=(3, 3),
                stride=(2, 2),
                padding=(1, 1)),
        InceptionLayer([192, 96, 208, 16, 48, 64], name='inception 4a'),
        InceptionLayer([160, 112, 224, 24, 64, 64], name='inception 4b'),
        InceptionLayer([128, 128, 256, 24, 64, 64], name='inception 4c'),
        InceptionLayer([112, 144, 288, 32, 64, 64], name='inception 4d'),
        InceptionLayer([256, 160, 320, 32, 28, 128], name='inception 4e'),
        MaxPool(poolsize=(3, 3),
                stride=(2, 2),
                padding=(1, 1)),
        InceptionLayer([256, 160, 320, 32, 128, 128], name='inception 5a'),
        InceptionLayer([384, 192, 384, 48, 128, 128], name='inception 5b'),
        AvgPool(poolsize=(7, 7),
                stride=(1, 1)),
        Dropout(0.4),
        FullyConnectedLayer(1000),
        Softmax(),
    ])
    if trained:
        googlenet.set_params(weights)
    return googlenet
