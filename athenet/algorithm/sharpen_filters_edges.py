"""
    Sharpens edges in convolutional layers' filters and removes other weights
    if edges was found

    Place for some description
"""

import numpy as np
from athenet.layers import ConvolutionalLayer

def sharpen_filter_edges(network):
    convolutional_layers = []
    for i in xrange(network.weighted_layers):
        if isinstance(network.weighted_layers[i], ConvolutionalLayer):
            convolutional_layers.append[i]

    for i in xrange(len(convolutional_layers)):
        layer = network.weighted_layers[convolutional_layers[i]]
        W = layer.W
        for x in xrange(W.shape[0]):
            for y in xrange(W.shape[1]):
                