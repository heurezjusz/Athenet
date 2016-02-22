"""Main Network class."""

import timeit
import numpy as np

import theano
import theano.tensor as T

from athenet.layers import WeightedLayer, ConvolutionalLayer
from athenet.data_loader import DataType
from athenet.utils import overwrite, save_data_to_pickle


class Network(object):
    """Neural network."""
    def __init__(self, layers):
        """Create neural network.

        layers: List of network's layers.
        """
        self._batch_size = None
        self._data_loader = None
        self.answers = None
        self.get_output = None

        self.snapshot_name = 'network'
        self.snapshot_interval = 1000
        self.verbosity = 1
        self._batch_index = T.lscalar()
        self._input = T.tensor4()
        self._correct_answers = T.ivector()
        self.layers = layers

        self.weighted_layers = [layer for layer in self.layers
                                if isinstance(layer, WeightedLayer)]
        self.convolutional_layers = [layer for layer in self.weighted_layers
                                     if isinstance(layer, ConvolutionalLayer)]

        self.batch_size = 1

    @property
    def data_loader(self):
        """Instance of class athenet.utils.DataLoader."""
        return self._data_loader

    @data_loader.setter
    def data_loader(self, value):
        self._data_loader = value
        if value:
            self.data_loader.batch_size = self.batch_size

    @property
    def batch_size(self):
        """Minibatch size."""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        if self._batch_size == value:
            return
        self._batch_size = value
        if self.data_loader:
            self.data_loader.batch_size = value

        for layer in self.convolutional_layers:
            layer.batch_size = self.batch_size
        self.layers[0].input = self._input
        for i in xrange(1, len(self.layers)):
            self.layers[i].input_layer = self.layers[i-1]
        self.layers[-1].set_cost(self._correct_answers)

        output = self.layers[-1].output
        self.answers = T.argsort(-output, axis=1)
        self.get_output = theano.function(
            inputs=[self._input],
            outputs=[output.flatten(1), self.answers.flatten(1)]
        )

    def test_accuracy(self, top_range=1):
        """Return network's accuracy on the test data.

        :top_range: Number or list representing top ranges to be used.
                    Network's answer is considered correct if correct answer
                    is among top_range most probable answers given by the
                    network.
        :return: Number or list representing network accuracy for given top
                 ranges.
        """
        return self._get_accuracy(top_range, DataType.test_data)

    def val_accuracy(self, top_range=1):
        """Return network's accuracy on the validation data.

        :top_range: Number or list representing top ranges to be used.
                    Network's answer is considered correct if correct answer
                    is among top_range most probable answers given by the
                    network.
        :return: Number or list representing network accuracy for given top
                 ranges.
        """
        return self._get_accuracy(top_range, DataType.validation_data)

    def _get_accuracy(self, top_range, data_type):
        return_list = isinstance(top_range, list)
        if not return_list:
            top_range = [top_range]
        max_top_range = max(top_range)

        expanded = self._correct_answers.dimshuffle(0, 'x')
        expanded = expanded.repeat(max_top_range, axis=1)
        eq = T.eq(expanded, self.answers[:, :max_top_range])
        get_accuracy = theano.function(
            inputs=[self._batch_index],
            outputs=[T.any(eq[:, :top], axis=1).mean() for top in top_range],
            givens={
                self._input:
                    self.data_loader.input(self._batch_index, data_type),
                self._correct_answers:
                    self.data_loader.output(self._batch_index, data_type)
            },
        )

        n_batches = self.data_loader.n_batches(data_type)
        accuracy = np.zeros(shape=(n_batches, len(top_range)))
        interval = n_batches/10
        if interval == 0:
            interval = 1
        for batch_index in xrange(n_batches):
            self.data_loader.load_data(batch_index, data_type)
            accuracy[batch_index, :] = np.asarray(get_accuracy(batch_index))
            if self.verbosity >= 3 or \
                    (self.verbosity >= 2 and batch_index % interval == 0):
                partial_accuracy = accuracy[:batch_index+1, :].mean(axis=0)
                text = ''
                for a in partial_accuracy:
                    text += ' {:.2f}%'.format(100*a)
                overwrite('{}/{} minibatches accuracy:{}'
                          .format(batch_index+1, n_batches, text))
        overwrite()

        accuracy = accuracy.mean(axis=0).tolist()
        if not return_list:
            return accuracy[0]
        return accuracy

    def get_params(self):
        """Return list of network's weights and biases.

        :return: List of pairs (W, b).
        """
        params = []
        for layer in self.weighted_layers:
            params += [(layer.W, layer.b)]
        return params

    def set_params(self, params):
        """Set network's weights and biases.

        :params: List of pairs (W, b).
        """
        for p, layer in zip(params, self.weighted_layers):
            layer.W = p[0]
            layer.b = p[1]

    def save_to_file(self, filename):
        """Save network's weights to file.

        :filename:Name of the file.
        """
        save_data_to_pickle(self.get_params(), filename)

    def evaluate(self, net_input):
        """Return network output for a given input.

        Batch size must be equal 1 to use this method. If it isn't, it will be
        set to 1.

        :net_input: Input for the network.
        :return: A pair consisting of list of probabilities for every answer
                 index and list of answer indexes sorted by their
                 probabilities descending.
        """
        self.batch_size = 1
        net_input = np.asarray(net_input, dtype=theano.config.floatX)
        n_channels, height, width = net_input.shape
        net_input = np.resize(net_input, (1, n_channels, height, width))
        return self.get_output(net_input)

    def train(self, n_epochs, learning_rate, momentum=0., weight_decay=0.,
              batch_size=None):
        """Train the network.

        :n_epochs: Number of epochs.
        :learning_rate: Learning rate.
        :momentum: Momentum coefficient.
        :weight_decay: Weight decay coefficient for L2 regularization.
        :batch_size: Size of minibatch to be set. If None then batch size that
                     is currenty set will be used.
        """
        if self.data_loader is None:
            raise Exception('data loader is not set')
        if not self.data_loader.train_data_available:
            raise Exception('training data not available')
        if batch_size is not None:
            self.batch_size = batch_size

        cost = self.layers[-1].cost
        weights = [layer.W_shared for layer in self.weighted_layers]
        biases = [layer.b_shared for layer in self.weighted_layers]
        weights_grad = T.grad(cost, weights)
        biases_grad = T.grad(cost, biases)

        if momentum:
            for layer in self.weighted_layers:
                layer.alloc_velocity()
            weights_vel = [layer.W_velocity for layer in self.weighted_layers]
            biases_vel = [layer.b_velocity for layer in self.weighted_layers]

            weights_vel_updates = [
                (v, momentum*v - weight_decay*learning_rate*w -
                    learning_rate*der)
                for w, v, der in zip(weights, weights_vel, weights_grad)]
            biases_vel_updates = [(v, momentum*v - learning_rate*der)
                                  for v, der in zip(biases_vel, biases_grad)]

            weights_updates = [(w, w + v)
                               for w, v in zip(weights, weights_vel)]
            biases_updates = [(b, b + v)
                              for b, v in zip(biases, biases_vel)]
            updates = weights_vel_updates + weights_updates + \
                biases_vel_updates + biases_updates
        else:
            weights_updates = [
                (w, (1. - weight_decay*learning_rate)*w - learning_rate*der)
                for w, der in zip(weights, weights_grad)]
            biases_updates = [(b, b - learning_rate*der)
                              for b, der in zip(biases, biases_grad)]
            updates = weights_updates + biases_updates

        train_model = theano.function(
            inputs=[self._batch_index],
            outputs=cost,
            updates=updates,
            givens={
                self._input:
                    self.data_loader.train_input(self._batch_index),
                self._correct_answers:
                    self.data_loader.train_output(self._batch_index)
            },
        )

        val_interval = self.data_loader.n_train_batches
        iteration = 0
        if self.verbosity >= 1:
            print 'Training for {} epochs with\n'\
                  'learning rate = {}, momentum = {}, weight decay = {}, '\
                  'minibatch size = {}'\
                  .format(n_epochs, learning_rate, momentum, weight_decay,
                          self.batch_size)
            print '{} minibatches per epoch'\
                  .format(self.data_loader.n_train_batches)
            start_time = timeit.default_timer()

        for epoch in xrange(1, n_epochs+1):
            if self.verbosity >= 1:
                print 'Epoch {}'.format(epoch)
                epoch_start_time = timeit.default_timer()
            for batch_index in xrange(self.data_loader.n_train_batches):
                self.data_loader.load_train_data(batch_index)
                train_model(batch_index)
                iteration += 1
                if self.snapshot_interval and \
                        iteration % self.snapshot_interval == 0:
                    self.save_to_file('{}_iteration_{}.pkl.gz'
                                      .format(self.snapshot_name, iteration))
                if self.data_loader.val_data_available:
                    if iteration % val_interval == 0:
                        accuracy = self.val_accuracy()
                        if self.verbosity >= 1:
                            print '\tAccuracy on validation data: {:.2f}%'\
                                  .format(100*accuracy)
            if self.verbosity >= 1:
                epoch_end_time = timeit.default_timer()
                print '\tTime: {:.1f}s'.format(
                    epoch_end_time - epoch_start_time)

        if self.verbosity >= 1:
            end_time = timeit.default_timer()
            print 'Training time: {:.1f}s'.format(end_time - start_time)
        if momentum:
            for layer in self.weighted_layers:
                layer.free_velocity()
