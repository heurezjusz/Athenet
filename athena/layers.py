"""Layers implementation."""

import numpy as np

import theano
import theano.tensor as T
from theano.tensor.nnet import conv, softmax
from theano.tensor.signal import downsample

class Layer(object):
    """Base class for network layer."""
    def __init__(self):
        self._input = None
        self.output = None
        self.cost = None

    @property
    def input(self):
        """Return layer input."""
        return self._input

    @input.setter
    def input(self, value):
        """Set layer's input and output variables."""
        self._input = value
        self.output = self._input

class Softmax(Layer):
    """Softmax layer."""

    @Layer.input.setter
    def input(self, value):
        """Set layer's input and output variables."""
        self._input = value
        self.output = softmax(self.input)

    def set_cost(self, y):
        """
        Set layer's cost variables.

        y: Desired output
        """
        self.cost = T.mean(-T.log(self.output)[T.arange(y.shape[0]), y])


class Activation(Layer):
    """Layer applying activation function to neurons."""
    def __init__(self, activation_function):
        """Create activation layer.

        activation_function: Activation function to be applied
        """
        super(Activation, self).__init__()
        self.activation_function = activation_function

    @Layer.input.setter
    def input(self, value):
        """Set layer's input and output variables."""
        self._input = value
        self.output = self.activation_function(self.input)


def relu(x):
    """Rectified linear activation function.

    x: Neuron input
    """
    return T.maximum(0., x)


class ReLU(Activation):
    """Layer applying rectified linear activation function."""
    def __init__(self):
        super(ReLU, self).__init__(relu)


class MaxPool(Layer):
    """Max-pooling layer."""
    def __init__(self, poolsize, stride=None):
        """Create max-pooling layer.

        poolsize: Pooling factor in the format (height, width)
        """
        super(MaxPool, self).__init__()
        self.poolsize = poolsize
        self.stride = stride

    @Layer.input.setter
    def input(self, value):
        """Set layer's input and output variables.

        value: Input in the format (batch size, number of channels,
                                    image height, image width)
        """
        self._input = value
        self.output = downsample.max_pool_2d(
            input=self.input,
            ds=self.poolsize,
            ignore_border=True,
            st=self.stride
        )


class LRN(Layer):
    """Local Response Normalization layer."""
    def __init__(self, local_range=5, k=1, alpha=0.0005, beta=0.75):
        """Create Local Response Normalization layer.

        local_range: Local channel range. Should be odd,
                     otherwise it will be incremented.
        k: Additive constant
        alpha: The scaling parameter
        beta: The exponent
        """
        super(LRN, self).__init__()
        if local_range % 2 == 0:
            local_range += 1
        self.local_range = local_range
        self.k = k
        self.alpha = alpha
        self.beta = beta

    @Layer.input.setter
    def input(self, value):
        """Set layer's input and output variables.

        value: Input in the format (batch size, number of channels,
                                    image height, image width)
        """
        self._input = value

        half = self.local_range // 2
        sq = T.sqr(self.input)
        bs, n_channels, h, w = self.input.shape
        extra_channels = T.alloc(0., bs, n_channels + 2*half, h, w)
        sq = T.set_subtensor(extra_channels[:, half:half+n_channels, :, :], sq)

        local_sums = 0
        for i in xrange(self.local_range):
            local_sums += sq[:, i:i+n_channels, :, :]

        self.output = self.input / (
            self.k + self.alpha/self.local_range * local_sums)**self.beta


class Dropout(Layer):
    """Dropout layer."""
    def __init__(self, p_dropout):
        """Create dropout layer.

        p_dropout: Weight dropout probability
        """
        super(Dropout, self).__init__()
        self.p_dropout = p_dropout

    @Layer.input.setter
    def input(self, value):
        """Set layer's input and output variables."""
        self._input = value
        self.output = (1. - self.p_dropout) * self.input


class WeightedLayer(Layer):
    """Layer with weights and biases."""
    def __init__(self, weights=None, biases=None):
        """Create weighted layer.

        weights: Array of weights's values
        biases: Array of biases' values
        """
        super(WeightedLayer, self).__init__()
        self.W_shared = None
        self.b_shared = None
        self.params = None

        if weights:
            self.W_shared = theano.shared(weights)
        if biases:
            self.b_shared = theano.shared(biases)

    @property
    def W(self):
        """Return copy of the layer's weights.

        return: Array of weights' values
        """
        return self.W_shared.get_value()

    @W.setter
    def W(self, value):
        """Set the layer's weights.

        value: Array of weights' alues
        """
        self.W_shared.set_value(value)

    @property
    def b(self):
        """Return copy of the layer's biases.

        return: Array of biases' values
        """
        return self.b_shared.get_value()

    @b.setter
    def b(self, value):
        """Set the layer's biases.

       value: Array of biases' values
        """
        self.b_shared.set_value(value)


class FullyConnectedLayer(WeightedLayer):
    """Fully connected layer."""
    def __init__(self, n_in, n_out):
        """Create fully connected layer.

        n_in: Number of input neurons
        n_out: Number of output neurons
        """
        super(FullyConnectedLayer, self).__init__()
        if not self.W_shared:
            W_value = np.asarray(
                np.random.normal(
                    loc=0.,
                    scale=np.sqrt(1. / n_out),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            self.W_shared = theano.shared(W_value, borrow=True)

        if not self.b_shared:
            b_value = np.zeros((n_out,), dtype=theano.config.floatX)
            self.b_shared = theano.shared(b_value, borrow=True)

        self.params = [self.W_shared, self.b_shared]

    @Layer.input.setter
    def input(self, value):
        """Set layer's input and output variables.

        value: Input in the format (n_in, n_out) or compatible
        """
        self._input = value.flatten(2)
        self.output = T.dot(self.input, self.W_shared) + self.b_shared


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

    @Layer.input.setter
    def input(self, value):
        """Set layer's input and output variables.

        value: Input in the format (batch size, number of channels,
                                    image height, image width) or compatible
        """
        self._input = value.reshape(self.image_shape)
        self.output = conv.conv2d(
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
