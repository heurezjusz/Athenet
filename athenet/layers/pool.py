"""Pooling layer."""

from theano.tensor.signal.pool import pool_2d

from athenet.layers import Layer


class PoolingLayer(Layer):
    """Pooling layer."""
    def __init__(self, poolsize, stride=None, mode='max', padding=(0, 0),
                 input_layer_name=None, name='pool'):
        """Create pooling layer.

        :poolsize: Shape of pooling filter in the format (height, width).
        :stride: Pair representing interval at which to apply the filters.
                 If None, then stride of the size of the pooling filter will be
                 used.
        :padding: Pair representing number of zero-valued pixels to add on
                  each side of the input.
        :mode: Pooling method: 'max' or 'avg'. Default 'max'.
        """
        super(PoolingLayer, self).__init__(input_layer_name, name)
        self.poolsize = poolsize
        if stride is None:
            self.stride = poolsize
        else:
            self.stride = stride
        self.padding = padding
        self.mode = mode

    @property
    def output_shape(self):
        image_h, image_w, n_channels = self.input_shape
        pad_h, pad_w = self.padding
        image_h += 2*pad_h
        image_w += 2*pad_w
        pool_h, pool_w = self.poolsize
        if self.stride is not None:
            stride_h, stride_w = self.stride
        else:
            stride_h, stride_w = pool_h, pool_w

        output_h = (image_h - pool_h) / stride_h + 1
        output_w = (image_w - pool_w) / stride_w + 1
        return (output_h, output_w, n_channels)

    def _get_output(self, layer_input):
        """Return layer's output.

        layer_input: Input in the format (batch size, number of channels,
                                          image height, image width).
        :return: Layer output.
        """
        if self.stride == self.poolsize:
            stride = None
        else:
            stride = self.stride
        if self.mode == 'avg':
            mode = 'average_exc_pad'
        else:
            mode = self.mode

        return pool_2d(
            input=layer_input,
            ds=self.poolsize,
            ignore_border=True,
            st=stride,
            padding=self.padding,
            mode=mode,
        )


class MaxPool(PoolingLayer):
    def __init__(self, poolsize, stride=None, padding=(0, 0),
                 input_layer_name=None, name='max_pool'):
        """Create max-pooling layer."""
        super(MaxPool, self).__init__(poolsize, stride, 'max', padding,
                                      input_layer_name, name)


class AvgPool(PoolingLayer):
    def __init__(self, poolsize, stride=None, padding=(0, 0),
                 input_layer_name=None, name='avg_pool'):
        """Create average-pooling layer."""
        super(AvgPool, self).__init__(poolsize, stride, 'avg', padding,
                                      input_layer_name, name)
