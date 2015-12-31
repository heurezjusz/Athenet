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


class MNISTDataLoader(DataLoader):
    """MNIST data loader."""
    def load_mnist_data(cls, filename=_MNIST_FILENAME, url=_MNIST_ORIGIN):
        """Load MNIST data from file.

        filename: Name of the file with MNIST data.
        url: Url for downloading MNIST data.
        return: List of training, validation and test data in the format
                (input, output).
        """
        train_set, valid_set, test_set = load_data(filename, url)

        test_set_in, test_set_out = cls._mnist_shared_dataset(test_set)
        valid_set_in, valid_set_out = cls._mnist_shared_dataset(valid_set)
        train_set_in, train_set_out = cls._mnist_shared_dataset(train_set)

        return [(train_set_in, train_set_out), (valid_set_in, valid_set_out),
                (test_set_in, test_set_out)]

    def _mnist_shared_dataset(cls, data):
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

    def __init__(self):
        self._batch_size = None
        datasets = self.load_mnist_data()

        self.train_set_in, self.train_set_out = datasets[0]
        self.valid_set_in, self.valid_set_out = datasets[1]
        self.test_set_in, self.test_set_out = datasets[2]

        self.train_set_size =\
            self.train_set_in.get_value(borrow=True).shape[0]
        self.valid_set_size =\
            self.valid_set_in.get_value(borrow=True).shape[0]
        self.test_set_size =\
            self.test_set_in.get_value(borrow=True).shape[0]

        super(MNISTDataLoader, self).__init__()

    @property
    def batch_size(self):
        return self._batch_size
    
    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value

        self.n_train_batches = self.train_set_size / self.batch_size
        self.n_valid_batches = self.valid_set_size / self.batch_size
        self.n_test_batches = self.test_set_size / self.batch_size

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
