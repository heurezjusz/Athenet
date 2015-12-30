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
    @classmethod
    def load_mnist_data(cls, filename=_MNIST_FILENAME, url=_MNIST_ORIGIN):
        """Load MNIST data from file.

        filename: Name of the file with MNIST data.
        url: Url for downloading MNIST data.
        return: List of training, validation and test data in the format
                (x, y).
        """
        train_set, valid_set, test_set = load_data(filename, url)

        test_set_in, test_set_out = cls._mnist_shared_dataset(test_set)
        valid_set_in, valid_set_out = cls._mnist_shared_dataset(valid_set)
        train_set_in, train_set_out = cls._mnist_shared_dataset(train_set)

        return [(train_set_in, train_set_out), (valid_set_in, valid_set_out),
                (test_set_in, test_set_out)]

    @classmethod
    def _mnist_shared_dataset(cls, data):
        """Create shared variables from given data.

        data: Data consisting of pairs (x, y).
        return: Theano shared variables created from data.
        """
        data_x, data_y = data
        data_x = np.resize(data_x, (data_x.shape[0], 1, 28, 28))
        shared_x = theano.shared(np.asarray(data_x,
                                            dtype=theano.config.floatX),
                                 borrow=True)
        shared_y = theano.shared(np.asarray(data_y,
                                            dtype=theano.config.floatX),
                                 borrow=True)
        return shared_x, T.cast(shared_y, 'int32')

    def __init__(self):
        super(MNISTDataLoader, self).__init__()
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

    @property
    def n_train_batches(self):
        """Return number of training minibatches."""
        return self.train_set_size / self.batch_size

    @property
    def n_valid_batches(self):
        """Return number of validation minibatches."""
        return self.valid_set_size / self.batch_size

    @property
    def n_test_batches(self):
        """Return number of testing minibatches."""
        return self.test_set_size / self.batch_size
