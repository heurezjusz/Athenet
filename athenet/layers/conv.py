"""Convolutional layer."""

import numpy as np

import theano
import theano.tensor as T

from athenet.layers import WeightedLayer
from athenet.utils.misc import convolution


class ConvolutionalLayer(WeightedLayer):
    """Convolutional layer."""
    def __init__(self, filter_shape, image_shape=None, stride=(1, 1),
                 padding=(0, 0), n_groups=1, batch_size=1):
        """Create convolutional layer.

        :filter_shape: Filter shape in the format
                       (filter height, filter width, number of filters).
        :image_shape: Image shape in the format
                      (image height, image width, number of channels).
        :stride: Pair representing interval at which to apply the filters.
        :padding: Pair representing number of zero-valued pixels to add on
                  each side of the input.
        :n_groups: Number of groups input and output channels will be split
                   into. Two channels are connected only if they belong to the
                   same group.
        :batch_size: Minibatch size.
        """
        super(ConvolutionalLayer, self).__init__()
        self._image_shape = None

        self.filter_shape = filter_shape
        self.stride = stride
        self.padding = padding
        self.n_groups = n_groups
        self.batch_size = batch_size
        self.image_shape = image_shape

    @property
    def image_shape(self):
        return self._image_shape

    @image_shape.setter
    def image_shape(self, value):
        if not value or self._image_shape == value:
            return
        self._image_shape = value

        h, w, n_filters = self.filter_shape
        conv_filter_shape = (n_filters, self.image_shape[2]/self.n_groups,
                             h, w)

        n_out = self.filter_shape[0] * np.prod(self.filter_shape[2:])
        W_value = np.asarray(
            np.random.normal(
                loc=0.,
                scale=np.sqrt(1. / n_out),
                size=conv_filter_shape
            ),
            dtype=theano.config.floatX
        )
        self.W_shared = theano.shared(W_value, borrow=True)

        b_value = np.zeros((n_filters,), dtype=theano.config.floatX)
        self.b_shared = theano.shared(b_value, borrow=True)

    @property
    def input_shape(self):
        return self.image_shape

    @input_shape.setter
    def input_shape(self, value):
        self.image_shape = value

    @property
    def output_shape(self):
        image_h, image_w, n_channels = self.image_shape
        pad_h, pad_w = self.padding
        image_h += 2 * pad_h
        image_w += 2 * pad_w
        filter_h, filter_w, n_filters = self.filter_shape
        stride_h, stride_w = self.stride

        output_h = (image_h - filter_h) / stride_h + 1
        output_w = (image_w - filter_w) / stride_w + 1
        return (output_h, output_w, n_filters)

    def _reshape_input(self, raw_layer_input):
        """Return input in the correct format for convolutional layer.

        :raw_layer_input: Input in the format (batch size, number of channels,
                                               image height, image width) or
                          compatible.
        """
        h, w, n_channels = self.image_shape
        conv_image_shape = (self.batch_size, n_channels, h, w)
        reshaped_input = raw_layer_input.reshape(conv_image_shape)

        pad_h, pad_w = self.padding
        h_in = h + 2*pad_h
        w_in = w + 2*pad_w

        extra_pixels = T.alloc(np.array(0., dtype=theano.config.floatX),
                               self.batch_size, n_channels, h_in, w_in)
        extra_pixels = T.set_subtensor(
            extra_pixels[:, :, pad_h:pad_h+h, pad_w:pad_w+w], reshaped_input)
        return extra_pixels

    def _get_output(self, layer_input):
        # By default, Theano doesn't use cuDNN convolutions if
        # subsample != (1, 1), so we need to call it manually
        conv_output = convolution(self.input, self.W_shared, self.stride,
                                  self.n_groups, self.image_shape,
                                  self.padding, self.batch_size,
                                  self.filter_shape)
        return conv_output + self.b_shared.dimshuffle('x', 0, 'x', 'x')
