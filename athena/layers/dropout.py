"""Dropout layer."""

import numpy as np

import theano
import theano.tensor as T

from athena.layers import Layer


class Dropout(Layer):
    """Dropout layer."""
    def __init__(self, p_dropout):
        """Create dropout layer.

        p_dropout: Weight dropout probability
        """
        super(Dropout, self).__init__()
        self.p_dropout = p_dropout

    def _get_output(self, layer_input):
        """Return layer's output.

        layer_input: Layer input.
        """
        return (1. - self.p_dropout) * layer_input

    def _get_train_output(self, layer_input):
        """Return layer's output.

        layer_input: Layer input.
        """
        pass  # TODO
