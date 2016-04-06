"""Normalization layer."""

import theano
import theano.tensor as T

from athenet.layers import Layer


class LRN(Layer):
    """Local Response Normalization layer."""
    def __init__(self, local_range=5, k=1, alpha=0.0002, beta=0.75,
                 input_layer_name=None, name='lrn'):
        """Create Local Response Normalization layer.

        :param local_range: Local channel range. Should be odd, otherwise it
                            will be incremented.
        :param k: Additive constant.
        :param alpha: The scaling parameter.
        :param beta: The exponent.
        """
        super(LRN, self).__init__(input_layer_name, name)
        if local_range % 2 == 0:
            local_range += 1
        self.local_range = local_range
        self.k = k
        self.alpha = alpha
        self.beta = beta

    def _get_output(self, layer_input):
        """Return layer's output.

        :param layer_input: Input in the format
                            (batch size, number of channels,
                             image height, image width).
        :return: Layer output.
        """
        half = self.local_range / 2
        sq = T.sqr(layer_input)
        bs, n_channels, h, w = layer_input.shape
        extra_channels = T.alloc(0., bs, n_channels + 2*half, h, w)
        sq = T.set_subtensor(extra_channels[:, half:half+n_channels, :, :], sq)

        local_sums = T.zeros_like(layer_input, dtype=theano.config.floatX)
        for i in xrange(self.local_range):
            local_sums += sq[:, i:i+n_channels, :, :]

        return layer_input / (
            self.k + self.alpha/self.local_range * local_sums)**self.beta
