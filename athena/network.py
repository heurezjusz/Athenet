"""Main Network class."""

import timeit
import numpy as np

import theano
import theano.tensor as T

from athena.layers import WeightedLayer, ConvolutionalLayer


class Network(object):
    """Neural network."""

    # Early stopping parameters
    initial_patience = 10000
    patience_increase = 2

    def __init__(self, layers, batch_size=1):
        """Create neural network.

        layers: List of network's layers
        batch_size: Minibatch size
        """
        self.output = None
        self.train_output = None
        self.y_out = None
        self.n_train_batches = None
        self.n_valid_batches = None
        self.n_test_batches = None
        self.train_set_x = None
        self.train_set_y = None
        self.valid_set_x = None
        self.valid_set_y = None
        self.test_set_x = None
        self.test_set_y = None
        self.get_output = None

        self._data_accuracy = None
        self._test_data_accuracy = None
        self._validation_data_accuracy = None
        self._datasets = None
        self._batch_size = None

        self.layers = layers
        self.x = T.tensor4('x')
        self.y = T.ivector('y')
        self._batch_index = T.lscalar()

        self.weighted_layers = [layer for layer in self.layers
                                if isinstance(layer, WeightedLayer)]
        self.convolutional_layers = [layer for layer in self.weighted_layers
                                     if isinstance(layer, ConvolutionalLayer)]

        self.params = []
        for layer in self.weighted_layers:
            self.params += layer.params

        self.batch_size = batch_size

    @property
    def batch_size(self):
        """Return batch size."""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        """Set batch size."""
        if self._batch_size == value:
            return
        self._batch_size = value
        for layer in self.convolutional_layers:
            layer.batch_size = self.batch_size

        self.layers[0].input = self.x
        for i in xrange(1, len(self.layers)):
            self.layers[i].input_layer = self.layers[i-1]
        self.output = self.layers[-1].output
        self.train_output = self.layers[-1].train_output
        self.y_out = T.argmax(self.output, axis=1)

        self._data_accuracy = T.mean(T.eq(self.y, self.y_out))
        self.get_output = theano.function(
            inputs=[self.x],
            outputs=self.output.flatten(1)
        )

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

    def get_params(self):
        """Return network's weights and biases.

        return: List of tuples (W, b)
        """
        params = []
        for layer in self.weighted_layers:
            params += [(layer.W, layer.b)]
        return params

    def set_params(self, params):
        """Set network's weights and biases.

        params: List of tuples (W, b)
        """
        for p, layer in zip(params, self.weighted_layers):
            layer.W = p[0]
            layer.b = p[1]

    def evaluate(self, x_in):
        """Return network output for a given input.

        x_in: Input for the network
        """
        self.batch_size = 1
        x_in = np.asarray(
            x_in,
            dtype=theano.config.floatX
        )
        height, width = x_in.shape
        x_in = np.resize(x_in, (1, 1, height, width))
        return self.get_output(x_in)

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
        if value:
            self.train_set_x, self.train_set_y = self.datasets[0]
            self.valid_set_x, self.valid_set_y = self.datasets[1]
            self.test_set_x, self.test_set_y = self.datasets[2]
        else:
            self.train_set_x, self.train_set_y = None, None
            self.valid_set_x, self.valid_set_y = None, None
            self.test_set_x, self.test_set_y = None, None

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

        grad = T.grad(cost, self.params)
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

        self.patience = self.initial_patience
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
        if self.datasets:
            self.n_train_batches = (self.train_set_x.get_value(borrow=True).
                                    shape[0] / self.batch_size)
            self.n_valid_batches = (self.valid_set_x.get_value(borrow=True).
                                    shape[0] / self.batch_size)
            self.n_test_batches = (self.test_set_x.get_value(borrow=True).
                                   shape[0] / self.batch_size)

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
        else:
            self.n_train_batches = None
            self.n_valid_batches = None
            self.n_test_batches = None

            self._validation_data_accuracy = None
            self._test_data_accuracy = None
