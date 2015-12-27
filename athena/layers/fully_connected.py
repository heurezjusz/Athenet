"""Fully connected layer."""

import numpy as np

import theano
import theano.tensor as T

from athena.layers import WeightedLayer


class FullyConnectedLayer(WeightedLayer):
    """Fully connected layer."""
    def __init__(self, n_in, n_out):
        """Create fully connected layer.

        n_in: Number of input neurons
        n_out: Number of output neurons
        """
        super(FullyConnectedLayer, self).__init__()

        W_value = np.asarray(
            np.random.normal(
                loc=0.,
                scale=np.sqrt(1. / n_out),
                size=(n_in, n_out)
            ),
            dtype=theano.config.floatX
        )
        self.W_shared = theano.shared(W_value, borrow=True)

        b_value = np.zeros((n_out,), dtype=theano.config.floatX)
        self.b_shared = theano.shared(b_value, borrow=True)

        self.params = [self.W_shared, self.b_shared]

    def _reshape_input(self, raw_layer_input):
        """Return input in the format that is suitable for this layer.

        raw_layer_input: Input in the format (n_in, n_out) or compatible.
        """
        return raw_layer_input.flatten(2)

    def _get_output(self, layer_input):
        """Return layer's output.

        layer_input: Layer input.
        """
        return T.dot(self.input, self.W_shared) + self.b_shared
