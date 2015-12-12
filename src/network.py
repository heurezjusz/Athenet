"""API for Athena, implemented in Theano."""

import timeit
import numpy as np

import theano
import theano.tensor as T
from theano.tensor.nnet import conv, softmax
from theano.tensor.signal import downsample


class Network(object):
    """Neural network."""

    patience = 10000
    patience_increase = 2

    def __init__(self, layers, batch_size=1):
        """Create neural network.

        layers: List of network's layers
        batch_size: Minibatch size
        """
        self.layers = layers
        self.x = T.matrix('x')
        self.y = T.ivector('y')
        self.output_function = lambda y: T.argmax(y, axis=1)
        self._batch_index = T.lscalar()
        self._datasets = None

        self.weighted_layers = [layer for layer in self.layers
                                if isinstance(layer, WeightedLayer)]
        self.convolutional_layers = [layer for layer in self.weighted_layers
                                     if isinstance(layer, ConvolutionalLayer)]

        self.params = []
        for layer in self.weighted_layers:
            self.params += layer.params

        self.batch_size = batch_size

        # fields used for training and testing
        self.n_train_batches = None
        self.n_valid_batches = None
        self.n_test_batches = None
        self.train_set_x = None
        self.train_set_y = None
        self.valid_set_x = None
        self.valid_set_y = None
        self.test_set_x = None
        self.test_set_y = None
        self._data_accuracy = None
        self._test_data_accuracy = None
        self._validation_data_accuracy = None

    @property
    def batch_size(self):
        """Return batch size."""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        """Set batch size."""
        self._batch_size = value
        for layer in self.convolutional_layers:
            layer.batch_size = self.batch_size

        self.layers[0].input = self.x
        for i in xrange(1, len(self.layers)):
            layer, prev_layer = self.layers[i], self.layers[i-1]
            layer.input = prev_layer.output
        self.output = self.layers[-1].output
        self.y_out = self.output_function(self.output)

        if self.datasets:
            self._update()

    def test_accuracy(self):
        """Return average network accuracy on the test data.

        Datasets must be set before using this method.

        return: A number between 0 and 1 representing average accuracy
        """
        test_accuracies = [self._test_data_accuracy(i) for i in
                           xrange(self.n_test_batches)]
        return np.mean(test_accuracies)

    @property
    def datasets(self):
        """Return datasets.

        return: Tuple containing training, validation and test data
        """
        return self._datasets

    @datasets.setter
    def datasets(self, value):
        """Load training data into network.

        value: Tuple containing training, validation and test data
        """
        self._datasets = value
        self.train_set_x, self.train_set_y = self.datasets[0]
        self.valid_set_x, self.valid_set_y = self.datasets[1]
        self.test_set_x, self.test_set_y = self.datasets[2]

        if self.batch_size:
            self._update()

    def train(self, learning_rate=0.1, n_epochs=100, batch_size=500,
              datasets=None):
        """Train and test the network.

        learning_rate: Learning rate
        n_epochs: Number of epochs
        batch_size: Size of minibatch
        datasets: Train, validation and test sets
        """
        self.batch_size = batch_size
        if datasets:
            self.datasets = datasets

        # set cost function for the last layer
        self.layers[-1].set_cost(self.y)
        cost = self.layers[-1].cost

        grad = [T.grad(cost, param) for param in self.params]
        updates = [(param, param - learning_rate*derivative)
                   for param, derivative in zip(self.params, grad)]

        train_model = theano.function(
            inputs=[self._batch_index],
            outputs=cost,
            updates=updates,
            givens={
                self.x: self.train_set_x[self._batch_index * batch_size:
                                         (self._batch_index+1) * batch_size],
                self.y: self.train_set_y[self._batch_index * batch_size:
                                         (self._batch_index+1) * batch_size]
            }
        )

        validation_interval = min(self.n_train_batches, self.patience / 2)
        best_validation_accuracy = 0.0
        epoch = 0
        iteration = 0
        done_looping = False

        start_time = timeit.default_timer()
        while (epoch < n_epochs) and (not done_looping):
            epoch += 1
            print 'Epoch {}'.format(epoch)
            for minibatch_index in xrange(self.n_train_batches):
                train_model(minibatch_index)
                iteration += 1
                if iteration % validation_interval == 0:
                    validation_accuracies = [
                        self._validation_data_accuracy(i)
                        for i in xrange(self.n_valid_batches)]
                    validation_accuracy = np.mean(validation_accuracies)
                    print '\tAccuracy on validation data: {:.2f}%'.format(
                        100 * validation_accuracy)
                    if validation_accuracy > best_validation_accuracy:
                        self.patience = max(self.patience, iteration *
                                            self.patience_increase)
                        best_validation_accuracy = validation_accuracy

                if self.patience <= iteration:
                    done_looping = True
                    break
        end_time = timeit.default_timer()

        print 'Accuracy on test data: {:.2f}%'.format(
            100 * self.test_accuracy())
        print 'Training time: {:.1f}s'.format(end_time - start_time)

    def _update(self):
        """Update fields that depend on both batch size and datasets."""
        self.n_train_batches = (self.train_set_x.get_value(borrow=True).
                                shape[0] / self.batch_size)
        self.n_valid_batches = (self.valid_set_x.get_value(borrow=True).
                                shape[0] / self.batch_size)
        self.n_test_batches = (self.test_set_x.get_value(borrow=True).
                               shape[0] / self.batch_size)

        self._data_accuracy = T.mean(T.eq(self.y, self.y_out))

        self._validation_data_accuracy = theano.function(
            inputs=[self._batch_index],
            outputs=self._data_accuracy,
            givens={
                self.x: self.valid_set_x[
                    self._batch_index * self.batch_size:
                    (self._batch_index+1) * self.batch_size
                ],
                self.y: self.valid_set_y[
                    self._batch_index * self.batch_size:
                    (self._batch_index+1) * self.batch_size
                ]
            }
        )
        self._test_data_accuracy = theano.function(
            inputs=[self._batch_index],
            outputs=self._data_accuracy,
            givens={
                self.x: self.test_set_x[
                    self._batch_index * self.batch_size:
                    (self._batch_index+1) * self.batch_size
                ],
                self.y: self.test_set_y[
                    self._batch_index * self.batch_size:
                    (self._batch_index+1) * self.batch_size
                ]
            }
        )


class Layer(object):
    """Base class for network layer."""
    def __init__(self):
        self._input = None
        self.output = None
        self.cost = None


class Softmax(Layer):
    """Softmax layer."""

    @property
    def input(self):
        """Return layer input."""
        return self._input

    @input.setter
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

    @property
    def input(self):
        """Return layer input."""
        return self._input

    @input.setter
    def input(self, value):
        """Set layer's input and output variables."""
        self._input = value
        self.output = self.activation_function(self.input)


def relu(x):
    """Rectified linear activation function.

    x: Neuron input
    """
    return T.maximum(0.0, x)


class ReLU(Activation):
    """Layer applying rectified linear activation function."""
    def __init__(self):
        super(ReLU, self).__init__(relu)


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

    @property
    def input(self):
        """Return layer input."""
        return self._input

    @input.setter
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
                       image height, image width)
        batch_size: Minibatch size
        """
        super(ConvolutionalLayer, self).__init__()
        self.image_size = image_size
        self.filter_shape = filter_shape
        self.batch_size = batch_size

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

    @property
    def input(self):
        """Return layer input."""
        return self._input

    @input.setter
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
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        """Set convolutional layer's minibatch size."""
        self._batch_size = value
        self.image_shape = (self.batch_size, self.filter_shape[1],
                            self.image_size[0], self.image_size[1])


class MaxPool(Layer):
    """Max-pooling layer."""
    def __init__(self, poolsize, stride=None):
        """Create max-pooling layer.

        poolsize: Pooling factor in the format (height, width)
        """
        super(MaxPool, self).__init__()
        self.poolsize = poolsize
        self.stride = stride

    @property
    def input(self):
        """Return layer input."""
        return self._input

    @input.setter
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

    @property
    def input(self):
        """Return layer input."""
        return self._input

    @input.setter
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
            self.k + self.alpha/self.local_range*local_sums)**self.beta
