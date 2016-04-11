"""Main Network class."""

import timeit
import numpy as np

import theano
import theano.tensor as T

from athenet.layers import WeightedLayer, ConvolutionalLayer, InceptionLayer, \
    FullyConnectedLayer
from athenet.utils import overwrite, save_data_to_pickle


class TrainConfig(object):
    """Training configuration.

    :n_epochs: Number of epochs.
    :batch_size: Number of minibatches (optional, can also be set in Network
                 class instance).
    :learning_rate: Learning rate.
    :momentum: Momentum. Default 0.
    :weight_decay: Weight decay. Default 0.
    :val_interval: Specifies number of ``val_interval_units`` between each
                   validation accuracy testing. Default 1.
    :val_interval_units: 'epochs' or 'batches'. Default 'epochs'.
    :lr_decay: Learning rate decay.
    :lr_decay_interval: Learning rate decay interval. If not None, then
                        learning rate will be multiplied by ``lr_decay`` after
                        each ``lr_decay_interval`` training units of type
                        ``lr_decay_interval_units``.
    :lr_decay_interval_units: 'epochs' or 'batches'. Default 'epochs'.
    :lr_decay_threshold: Threshold value for standard deviation of accuracy.
                         If None, then learning rate will be reduced when
                         standard deviation of accuracy is below threshold.
    :lr_decay_val_range: Number of previous iterations' accuracies to include
                         in standard deviation. Default 4.
    """
    def __init__(self):
        self.n_epochs = None
        self.batch_size = None

        self.learning_rate = None
        self.momentum = 0.
        self.weight_decay = 0.

        self.val_interval = 1
        self.val_interval_units = 'epochs'

        self.lr_decay = None
        self.lr_decay_interval = None
        self.lr_decay_interval_units = 'epochs'

        self.lr_decay_threshold = None
        self.lr_decay_val_range = 4

    def __str__(self):
        output = '{} epochs, minibatch size = {}, learning rate = {}'\
                 .format(self.n_epochs, self.batch_size, self.learning_rate)
        if self.momentum:
            output += ', momentum = {}'.format(self.momentum)
        if self.weight_decay:
            output += ', weight_decay = {}'.format(self.weight_decay)
        if self.lr_decay_interval is not None or \
                self.lr_decay_threshold is not None:
            output += '\nLearning rate decay = {}: '.format(self.lr_decay)
            if self.lr_decay_interval is not None:
                output += ', interval = {} {}'.format(
                    self.lr_decay_interval, self.lr_decay_interval_units)
            if self.lr_decay_threshold is not None:
                output += ', threshold = {}, validation range = {}'.format(
                    self.lr_decay_threshold, self.lr_decay_val_range)
        return output


class Network(object):
    """Neural network."""
    def __init__(self, layers):
        """Create neural network.

        :param layers: List of network's layers.
        """
        self._batch_size = None
        self._data_loader = None
        self.answers = None
        self.get_output = None
        self._accuracy = None
        self._accuracy_config = None

        self.snapshot_name = 'network'
        self.snapshot_interval = 0
        self.verbosity = 1
        self._batch_index = T.lscalar()
        if isinstance(layers[0], FullyConnectedLayer):
            self._input = T.matrix()
        elif isinstance(layers[0], ConvolutionalLayer):
            self._input = T.tensor4()
        else:
            raise Exception('{} is not supported as input layer'.format(
                type(layers[0])))
        self._correct_answers = T.ivector()
        self.layers = layers

        self.weighted_layers = [layer for layer in self.layers
                                if isinstance(layer, WeightedLayer) or
                                isinstance(layer, InceptionLayer)]
        self.convolutional_layers = [layer for layer in self.weighted_layers
                                     if isinstance(layer, ConvolutionalLayer)]

        self.layers_dict = {}
        for layer in self.layers:
            if layer.name is not None:
                self.layers_dict[layer.name] = layer
        self.batch_size = 1

    @property
    def data_loader(self):
        """Instance of :class:`athenet.data_loader.DataLoader`."""
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

        for layer in self.layers:
            layer.batch_size = self.batch_size
        self.layers[0].input = self._input
        self.layers[0].train_input = self._input
        for layer, prev_layer in zip(self.layers[1:], self.layers[:-1]):
            if layer.input_layer_name is None:
                layer.input_layer = prev_layer
            elif isinstance(layer.input_layer_name, list):
                layer.input_layers = [
                    self.layers_dict[layer_name]
                    for layer_name in layer.input_layer_names]
            else:
                layer.input_layer = self.layers_dict[layer.input_layer_name]

        output = self.layers[-1].output
        self.answers = T.argsort(-output, axis=1)
        self.get_output = theano.function(
            inputs=[self._input],
            outputs=[output.flatten(1), self.answers.flatten(1)]
        )

    def test_accuracy(self, top_range=1):
        """Return network's accuracy on the test data.

        :param top_range: Number or list representing top ranges to be used.
                          Network's answer is considered correct if correct
                          answer is among top_range most probable answers given
                          by the network.
        :return: Number or list representing network accuracy for given top
                 ranges.
        """
        return self._get_accuracy(top_range, 'test_data')

    def val_accuracy(self, top_range=1):
        """Return network's accuracy on the validation data.

        :param top_range: Number or list representing top ranges to be used.
                          Network's answer is considered correct if correct
                          answer is among top_range most probable answers given
                          by the network.
        :return: Number or list representing network accuracy for given top
                 ranges.
        """
        return self._get_accuracy(top_range, 'validation_data')

    def _get_accuracy(self, top_range, data_type):
        return_list = isinstance(top_range, list)
        if not return_list:
            top_range = [top_range]
        max_top_range = max(top_range)

        expanded = self._correct_answers.dimshuffle(0, 'x')
        expanded = expanded.repeat(max_top_range, axis=1)
        eq = T.eq(expanded, self.answers[:, :max_top_range])

        # Compile new function only if top range or data type has changed
        if self._accuracy_config != [top_range, data_type]:
            self._accuracy = theano.function(
                inputs=[self._batch_index],
                outputs=[T.any(eq[:, :top], axis=1).mean()
                         for top in top_range],
                givens={
                    self._input:
                        self.data_loader.input(self._batch_index, data_type),
                    self._correct_answers:
                        self.data_loader.output(self._batch_index, data_type)
                },
            )
            self._accuracy_config = [top_range, data_type]

        n_batches = self.data_loader.n_batches(data_type)
        accuracy = np.zeros(shape=(n_batches, len(top_range)))
        interval = n_batches/10
        if interval == 0:
            interval = 1
        for batch_index in xrange(n_batches):
            self.data_loader.load_data(batch_index, data_type)
            accuracy[batch_index, :] = np.asarray(self._accuracy(batch_index))
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

        :param params: List of pairs (W, b).
        """
        for layer, p in zip(self.weighted_layers, params):
            layer.set_params(p)

    def save_to_file(self, filename):
        """Save network's weights to file.

        :param filename: Name of the file.
        """
        save_data_to_pickle(self.get_params(), filename)

    def evaluate(self, net_input):
        """Return network output for a given input.

        Batch size must be equal 1 to use this method. If it isn't, it will be
        set to 1.

        :param net_input: Input for the network.
        :return: A pair consisting of list of probabilities for every answer
                 index and list of answer indexes sorted by their
                 probabilities descending.
        """
        self.batch_size = 1
        net_input = np.asarray(net_input, dtype=theano.config.floatX)
        net_input = np.resize(net_input, (1,)+net_input.shape)
        return self.get_output(net_input)

    def _convert_to_batches(self, interval, units):
        if interval is None:
            return None
        if units == 'epochs':
            return int(interval * self.data_loader.n_train_batches)
        return interval

    def _decrease_learning_rate(self, lr, lr_decay):
        new_lr = np.float32(lr.get_value() * lr_decay)
        lr.set_value(new_lr)
        return new_lr

    def train(self, config):
        """Train the network.

        :param config: Instance of :class:`TrainConfig`.
        """
        if self.data_loader is None:
            raise Exception('data loader is not set')
        if not self.data_loader.train_data_available:
            raise Exception('training data not available')
        if config.batch_size is not None:
            self.batch_size = config.batch_size

        val_interval = self._convert_to_batches(config.val_interval,
                                                config.val_interval_units)
        lr_decay_interval = self._convert_to_batches(
            config.lr_decay_interval,
            config.lr_decay_interval_units)

        self.layers[-1].set_cost(self._correct_answers)
        cost = self.layers[-1].cost
        lr = theano.shared(np.array(config.learning_rate,
                           dtype=theano.config.floatX))
        weights = [layer.W_shared for layer in self.weighted_layers]
        biases = [layer.b_shared for layer in self.weighted_layers]
        weights_grad = T.grad(cost, weights)
        biases_grad = T.grad(cost, biases)

        if config.momentum:
            momentum = theano.shared(np.array(config.momentum,
                                              dtype=theano.config.floatX))
            for layer in self.weighted_layers:
                layer.alloc_velocity()
            weights_vel = [layer.W_velocity for layer in self.weighted_layers]
            biases_vel = [layer.b_velocity for layer in self.weighted_layers]

            weights_vel_updates = [
                (v, momentum*v - config.weight_decay*lr*w - lr*der)
                for w, v, der in zip(weights, weights_vel, weights_grad)]
            biases_vel_updates = [(v, momentum*v - lr*der)
                                  for v, der in zip(biases_vel, biases_grad)]

            weights_updates = [(w, w + v)
                               for w, v in zip(weights, weights_vel)]
            biases_updates = [(b, b + v)
                              for b, v in zip(biases, biases_vel)]
            updates = weights_vel_updates + weights_updates + \
                biases_vel_updates + biases_updates
        else:
            weights_updates = [
                (w, (1. - config.weight_decay*lr)*w - lr*der)
                for w, der in zip(weights, weights_grad)]
            biases_updates = [(b, b - lr*der)
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

        iteration = 0
        if config.lr_decay_threshold is not None:
            prev_accuracies = np.zeros(config.val_range)
            pos = 0
            cycle_finished = False
        if self.verbosity >= 1:
            print 'Training:'
            print config
            print '{} minibatches per epoch'\
                  .format(self.data_loader.n_train_batches)
            start_time = timeit.default_timer()
        for epoch in xrange(1, config.n_epochs+1):
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
                if lr_decay_interval is not None and \
                        iteration % lr_decay_interval == 0:
                    new_lr = self._decrease_learning_rate(lr, config.lr_decay)
                    if self.verbosity >= 1:
                        print '\tLearning rate = {:.2f}'.format(new_lr)

                if not self.data_loader.val_data_available or \
                        iteration % val_interval != 0:
                    continue  # do not check validation accuracy
                accuracy = self.val_accuracy()
                if self.verbosity >= 1:
                    print '\tAccuracy on validation data: {:.2f}%'\
                          .format(100*accuracy)
                if config.lr_decay_threshold is not None:
                    prev_accuracies[pos] = 100*accuracy
                    pos = (pos + 1) % config.val_range
                    if cycle_finished:
                        lr_val = lr.get_value()
                        std = prev_accuracies.std() / lr_val
                        if self.verbosity >= 2:
                            print '\tstandard deviation = {}'.format(std)
                        if std < config.lr_decay_threshold:
                            new_lr = self._decrease_learning_rate(
                                lr, config.lr_decay)
                            if self.verbosity >= 1:
                                print '\tLearning rate = {:.2f}'.format(new_lr)
                            pos = 0
                            cycle_finished = False
                    elif pos == config.val_range - 1:
                        cycle_finished = True
            if self.verbosity >= 1:
                epoch_end_time = timeit.default_timer()
                print '\tTime: {:.1f}s'.format(
                    epoch_end_time - epoch_start_time)

        if self.verbosity >= 1:
            end_time = timeit.default_timer()
            print 'Training time: {:.1f}s'.format(end_time - start_time)
        if config.momentum:
            for layer in self.weighted_layers:
                layer.free_velocity()
