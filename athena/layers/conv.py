"""Convolutional layer."""

import numpy as np

import theano
from theano.tensor.nnet import conv

from athena.layers import WeightedLayer


class ConvolutionalLayer(WeightedLayer):
    """Convolutional layer."""
    def __init__(self, image_size, filter_shape, batch_size=1):
        """Create convolutional layer.

        image_size: Image size in the format (image height, image width)
        filter_shape: Shape of the filter in the format
                      (number of output channels, number of input channels,
                       filter height, filter width)
        batch_size: Minibatch size
        """
        super(ConvolutionalLayer, self).__init__()
        self.image_size = image_size
        self.filter_shape = filter_shape
        self.batch_size = batch_size
        self._batch_size = None
        self.image_shape = None

        if not self.W_shared:
            n_out = self.filter_shape[0] * np.prod(self.filter_shape[2:])
            W_value = np.asarray(
                np.random.normal(
                    loc=0.,
                    scale=np.sqrt(1. / n_out),
                    size=self.filter_shape
                ),
                dtype=theano.config.floatX
            )
            self.W_shared = theano.shared(W_value, borrow=True)

        if not self.b_shared:
            b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
            self.b_shared = theano.shared(b_values, borrow=True)

        self.params = [self.W_shared, self.b_shared]

    def _reshape_input(self, raw_layer_input):
        """Return input in the format that is suitable for this layer.

        raw_layer_input: Input in the format (batch size, number of channels,
                                              image height, image width)
                         or compatible.
        """
        return raw_layer_input.reshape(self.image_shape)

    def _get_output(self, layer_input):
        """Return layer's output.

        layer_input: Layer input.
        """
        return conv.conv2d(
            input=self.input,
            filters=self.W_shared,
            filter_shape=self.filter_shape,
            image_shape=self.image_shape
        ) + self.b_shared.dimshuffle('x', 0, 'x', 'x')

    @property
    def batch_size(self):
        "Return batch size."
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        """Set convolutional layer's minibatch size."""
        self._batch_size = value
        self.image_shape = (self.batch_size, self.filter_shape[1],
                            self.image_size[0], self.image_size[1])
