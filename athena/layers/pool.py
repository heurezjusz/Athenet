"""Pooling layer."""

from theano.tensor.signal import downsample

from athena.layers import Layer


class MaxPool(Layer):
    """Max-pooling layer."""
    def __init__(self, poolsize, stride=None):
        """Create max-pooling layer.

        poolsize: Pooling factor in the format (height, width)
        """
        super(MaxPool, self).__init__()
        self.poolsize = poolsize
        self.stride = stride

    def _get_output(self, layer_input):
        """Return layer's output.

        layer_input: Input in the format (batch size, number of channels,
                                          image height, image width).
        """
        return downsample.max_pool_2d(
            input=layer_input,
            ds=self.poolsize,
            ignore_border=True,
            st=self.stride
        )
