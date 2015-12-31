"""MNIST data loader."""

import os
import numpy as np

import theano
import theano.tensor as T

from athenet.utils import DataLoader, load_data


_MNIST_FILENAME = os.path.join(os.path.dirname(__file__),
                               '../../bin/mnist.pkl.gz')
_MNIST_ORIGIN = ('http://www.iro.umontreal.ca/~lisa/deep/data/mnist/'
                 'mnist.pkl.gz')


def _mnist_shared_dataset(data):
    """Create shared variables from given data.

    data: Data consisting of pairs (input, output).
    return: Theano shared variables created from data.
    """
    data_in, data_out = data
    data_in = np.resize(data_in, (data_in.shape[0], 1, 28, 28))
    shared_in = theano.shared(
        np.asarray(data_in, dtype=theano.config.floatX), borrow=True)
    shared_out = theano.shared(
        np.asarray(data_out, dtype=theano.config.floatX), borrow=True)
    return shared_in, T.cast(shared_out, 'int32')


class MNISTDataLoader(DataLoader):
    """MNIST data loader."""
    def __init__(self, filename=_MNIST_FILENAME, url=_MNIST_ORIGIN):
        train_set, valid_set, test_set = load_data(filename, url)

        self.test_set_in, self.test_set_out =\
            _mnist_shared_dataset(test_set)
        self.valid_set_in, self.valid_set_out =\
            _mnist_shared_dataset(valid_set)
        self.train_set_in, self.train_set_out =\
            _mnist_shared_dataset(train_set)

        self.train_set_size =\
            self.train_set_in.get_value(borrow=True).shape[0]
        self.valid_set_size =\
            self.valid_set_in.get_value(borrow=True).shape[0]
        self.test_set_size =\
            self.test_set_in.get_value(borrow=True).shape[0]

        super(MNISTDataLoader, self).__init__()

    def train_input(self, batch_index):
        return self._get_subset(self.train_set_in, batch_index)

    def train_output(self, batch_index):
        return self._get_subset(self.train_set_out, batch_index)

    def valid_input(self, batch_index):
        return self._get_subset(self.valid_set_in, batch_index)

    def valid_output(self, batch_index):
        return self._get_subset(self.valid_set_out, batch_index)

    def test_input(self, batch_index):
        return self._get_subset(self.test_set_in, batch_index)

    def test_output(self, batch_index):
        return self._get_subset(self.test_set_out, batch_index)
