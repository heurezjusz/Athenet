"""Concatenation layer."""

import numpy as np

import theano.tensor as T

from athenet.layers import Layer


class Concatenation(Layer):
    """Concatenation layer."""
    def __init__(self, input_layer_names=None, name='concat'):
        """Create concatenation layer.

        :input_layer_names: List of input layers' names.
        """
        super(Concatenation, self).__init__(input_layer_names, name)
        self.input_shapes = None
        self._input_layers = None

    @property
    def input_layer_names(self):
        return self.input_layer_name

    @property
    def input_layers(self):
        return self._input_layers

    @input_layers.setter
    def input_layers(self, input_layers):
        self._input_layers = input_layers
        self.input_shapes = [layer.output_shape for layer in input_layers]
        for input_shape in self.input_shapes[1:]:
            if input_shape[:-1] != self.input_shapes[0][:-1]:
                raise ValueError('all input layer image size must match')

        self.input = [layer.output for layer in input_layers]
        self.train_input = [layer.train_output for layer in input_layers]

    @property
    def output_shape(self):
        x, y = self.input_shapes[0][:2]
        n_channels = np.sum(
            [input_shape[2] for input_shape in self.input_shapes])
        return (x, y, n_channels)

    def _get_output(self, layer_inputs):
        return T.concatenate(layer_inputs, axis=1)
