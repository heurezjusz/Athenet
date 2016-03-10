"""Inception Layer."""

import numpy as np

import theano

from athenet.layers import Layer, ConvolutionalLayer, ReLU, MaxPool, \
    Concatenation


class InceptionLayer(object):
    def __init__(self, n_filters, input_layer_name=None, name='inception'):
        """Create inception layer.

        :n_filters: List of length 6: number of filters in convolutional
                    layers.
        """
        super(InceptionLayer, self).__init__(input_layer_name, name)

        layer_list1 = [
            ConvolutionalLayer(filter_shape=(n_filters[0], 1, 1)),
            ReLU(),
        ]
        layer_list2 = [
            ConvolutionalLayer(filter_shape=(n_filters[1], 1, 1)),
            ReLU(),
            ConvolutionalLayer(filter_shape=(n_filters[2], 3, 3),
                               padding=(1, 1)),
            ReLU(),
        ]
        layer_list3 = [
            ConvolutionalLayer(filter_shape=(n_filters[3], 1, 1)),
            ReLU(),
            ConvolutionalLayer(filter_shape=(n_filters[4], 5, 5),
                               padding=(2, 2)),
            ReLU(),
        ]
        layer_list4 = [
            MaxPool(poolsize=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1)),
            ConvolutionalLayer(filter_shape=(n_filters[5], 1, 1)),
            ReLU(),
        ]
        self.layer_lists = [layer_list1, layer_list2, layer_list3, layer_list4]
        self.bottom_layers = [layer_list[0] for layer_list in self.layer_lists]
        self.top_layers = [layer_list[-1] for layer_list in self.layer_lists]

        self.concat = Concatenation()
        self.concat.input_layers = top_layers

    @property
    def output_shape(self):
        return self.concat.output_shape

    @property
    def input_layer(self):
        return self._input_layer

    @input_layer.setter
    def input_layer(self, input_layer):
        self._input_layer = input_layer
        self.input_shape = input_layer.output_shape

        for bottom_layer in self.bottom_layers:
            bottom_layer.input_layer = input_layer
        # TODO: connect layers and set self.output
