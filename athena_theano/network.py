"""API for Athena, implemented in Theano."""

import timeit

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv, softmax
from theano.tensor.signal import downsample


class Network(object):
    """Neural network."""
    def __init__(self, layers, x, y, output_function=lambda y: y,
                 batch_size=1):
        """Create neural network.

        layers: List of layer from which the network is to be created
        x: Input format
        y: Output format
        output_function: Function mapping network output to desired output
        batch_size: Minibatch size
        """
        self.layers = layers
        self.x = x
        self.y = y
        self.output_function = output_function
        self.batch_index = T.lscalar()

        self.n_train_batches = None
        self.n_valid_batches = None
        self.n_test_batches = None
        self.train_set_x = None
        self.train_set_y = None
        self.valid_set_x = None
        self.valid_set_y = None
        self.test_set_x = None
        self.test_set_y = None
        self.validation_accuracy = None
        self.test_accuracy = None

        self.weighted_layers = [layer for layer in self.layers
                                if isinstance(layer, WeightedLayer)]
        self.convolutional_layers = [layer for layer in self.weighted_layers
                                     if isinstance(layer, ConvolutionalLayer)]

        self.params = []
        for layer in self.weighted_layers:
            self.params += layer.params

        self.set_batch_size(batch_size)

    def set_batch_size(self, batch_size):
        """Set batch size.

        batch_size: Miniatch size to be set
        """
        self.batch_size = batch_size
        for layer in self.convolutional_layers:
            layer.set_batch_size(self.batch_size)

        self.layers[0].set_input(self.x)
        for i in xrange(1, len(self.layers)):
            layer, prev_layer = self.layers[i], self.layers[i-1]
            layer.set_input(prev_layer.output)
        self.output = self.layers[-1].output
        self.y_out = self.output_function(self.output)

    def accuracy(self, y):
        """Return average network accuracy

        y: List of desired outputs
        return: A number between 0 and 1 representing average accuracy
        """
        return T.mean(T.eq(y, self.y_out))

    def set_training_data(self, datasets):
        """Load given training data into network.

        datasets: Train, validation and test sets
        """
        self.train_set_x, self.train_set_y = datasets[0]
        self.valid_set_x, self.valid_set_y = datasets[1]
        self.test_set_x, self.test_set_y = datasets[2]

        self.n_train_batches = (self.train_set_x.get_value(borrow=True).
                                shape[0] / self.batch_size)
        self.n_valid_batches = (self.valid_set_x.get_value(borrow=True).
                                shape[0] / self.batch_size)
        self.n_test_batches = (self.test_set_x.get_value(borrow=True).
                               shape[0] / self.batch_size)

        self.validate_accuracy = theano.function(
            inputs=[self.batch_index],
            outputs=self.accuracy(self.y),
            givens={
                self.x: self.valid_set_x[
                    self.batch_index * self.batch_size:
                    (self.batch_index+1) * self.batch_size
                ],
                self.y: self.valid_set_y[
                    self.batch_index * self.batch_size:
                    (self.batch_index+1) * self.batch_size
                ]
            }
        )
        self.test_accuracy = theano.function(
            inputs=[self.batch_index],
            outputs=self.accuracy(self.y),
            givens={
                self.x: self.test_set_x[
                    self.batch_index * self.batch_size:
                    (self.batch_index+1) * self.batch_size
                ],
                self.y: self.test_set_y[
                    self.batch_index * self.batch_size:
                    (self.batch_index+1) * self.batch_size
                ]
            }
        )

    def train(self, learning_rate=0.1, n_epochs=200, batch_size=500,
              datasets=None):
        """Train and test the network

        learning_rate: Learning rate
        n_epochs: Number of epochs
        batch_size: Size of minibatch
        datasets: Train, validation and test sets
        """
        self.set_batch_size(batch_size)

        if datasets:
            self.set_training_data(datasets)

        # set cost function for the last layer
        self.layers[-1].set_cost(self.y)
        cost = self.layers[-1].cost

        grad = [T.grad(cost, param) for param in self.params]
        updates = [(param, param - learning_rate*derivative)
                   for param, derivative in zip(self.params, grad)]

        train_model = theano.function(
            inputs=[self.batch_index],
            outputs=cost,
            updates=updates,
            givens={
                self.x: self.train_set_x[self.batch_index * batch_size:
                                         (self.batch_index+1) * batch_size],
                self.y: self.train_set_y[self.batch_index * batch_size:
                                         (self.batch_index+1) * batch_size]
            }
        )

        patience = 10000
        patience_increase = 2
        validation_interval = min(self.n_train_batches, patience / 2)
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
                    validation_accuracies = [self.validate_accuracy(i) for i
                                             in xrange(self.n_valid_batches)]
                    validation_accuracy = np.mean(validation_accuracies)
                    print '\tAccuracy on validation data: {:.2f}%'.format(
                        100 * validation_accuracy)
                    if validation_accuracy > best_validation_accuracy:
                        patience = max(patience, iteration * patience_increase)
                        best_validation_accuracy = validation_accuracy

                if patience <= iteration:
                    done_looping = True
                    break
        end_time = timeit.default_timer()

        test_accuracies = [self.test_accuracy(i) for i in
                           xrange(self.n_test_batches)]
        test_accuracy = np.mean(test_accuracies)
        print 'Accuracy on test data: {:.2f}%'.format(100 * test_accuracy)
        print 'Training time: {:.1f}s'.format(end_time - start_time)


class Layer(object):
    """Base class for network layer."""
    def __init__(self):
        self.input = None
        self.output = None
        self.cost = None


class Softmax(Layer):
    """Softmax layer."""
    def set_input(self, input):
        """Set layer's input and output variables.

        input: Variable to be set as layer's input
        """
        self.input = input
        self.output = softmax(input)

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

    def set_input(self, input):
        """Set layer's input and output variables.

        input: Variable to be set as layer's input
        """
        self.input = input
        self.output = self.activation_function(input)


def relu(x):
    """Rectified linear activation function

    x: Neuron input
    """
    return T.maximum(0.0, x)


class ReLU(Activation):
    """Layer applying rectified linear activation function."""
    def __init__(self):
        super(ReLU, self).__init__(relu)


class WeightedLayer(Layer):
    """Layer with weights and biases."""
    def __init__(self):
        super(WeightedLayer, self).__init__()
        self.W = None
        self.b = None


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
                loc=0.0,
                scale=np.sqrt(1.0 / n_out),
                size=(n_in, n_out)
            ),
            dtype=theano.config.floatX
        )
        self.W = theano.shared(W_value, borrow=True)

        b_value = np.zeros((n_out,), dtype=theano.config.floatX)
        self.b = theano.shared(b_value, borrow=True)

        self.params = [self.W, self.b]

    def set_input(self, input):
        """Set layer's input and output variables.

        input: Variable to be set as layer's input
        """
        self.input = input.flatten(2)
        self.output = T.dot(self.input, self.W) + self.b


class ConvolutionalLayer(WeightedLayer):
    """Convolutional layer."""
    def __init__(self, image_size, filter_shape, batch_size=1):
        """Create convolutional layer.

        image_size: Image size in the format (image height, image width)
        filter_shape: Shape of the filter in the format
                      (number of filters, number of input feature maps,
                       filter height, filter width)
        batch_size: Minibatch size
        """
        super(ConvolutionalLayer, self).__init__()
        self.image_size = image_size
        self.filter_shape = filter_shape
        self.set_batch_size(batch_size)

        n_out = self.filter_shape[0] * np.prod(self.filter_shape[2:])
        W_value = np.asarray(
            np.random.normal(
                loc=0.0,
                scale=np.sqrt(1.0 / n_out),
                size=self.filter_shape
            ),
            dtype=theano.config.floatX
        )
        self.W = theano.shared(W_value, borrow=True)

        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(b_values, borrow=True)

        self.params = [self.W, self.b]

    def set_input(self, input):
        """Set layer's input and output variables.

        input: Variable to be set as layer's input
        """
        self.input = input.reshape(self.image_shape)
        self.output = conv.conv2d(
            input=self.input,
            filters=self.W,
            filter_shape=self.filter_shape,
            image_shape=self.image_shape
        ) + self.b.dimshuffle('x', 0, 'x', 'x')

    def set_batch_size(self, batch_size):
        """Set convolutional layer's minibatch size.

        batch_size: Batch size
        """
        self.batch_size = batch_size
        self.image_shape = (self.batch_size, self.filter_shape[1],
                            self.image_size[0], self.image_size[1])


class MaxPool(Layer):
    """Max-pooling layer."""
    def __init__(self, poolsize):
        """Create max-pooling layer.

        poolsize: the pooling factor in the format (height, width)
        """
        super(MaxPool, self).__init__()
        self.poolsize = poolsize

    def set_input(self, input):
        """Set layer's input and output variables.

        input: Variable to be set as layer's input
        """
        self.input = input
        self.output = downsample.max_pool_2d(
            input=self.input,
            ds=self.poolsize,
            ignore_border=True
        )
