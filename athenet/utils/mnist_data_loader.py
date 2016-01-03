"""MNIST data loader."""

import numpy as np

import theano
import theano.tensor as T

from athenet.utils import DataLoader, load_data, get_bin_path


class MNISTDataLoader(DataLoader):
    """MNIST data loader."""

    MNIST_filename = 'mnist.pkl.gz'
    MNIST_origin = ('http://www.iro.umontreal.ca/~lisa/deep/data/mnist/'
                    'mnist.pkl.gz')

    @staticmethod
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

    def __init__(self, filename=MNIST_filename, url=MNIST_origin):
        """Create MNIST data loader.

        filename: Name of a file with MNIST data.
        url: Url for downloading MNIST data.
        """
        super(MNISTDataLoader, self).__init__()

        train_set, val_set, test_set = load_data(get_bin_path(filename), url)

        self.test_in, self.test_out = self._mnist_shared_dataset(test_set)
        self.val_in, self.val_out = self._mnist_shared_dataset(val_set)
        self.train_in, self.train_out = self._mnist_shared_dataset(train_set)

        self.train_set_size = self.train_in.get_value(borrow=True).shape[0]
        self.val_set_size = self.val_in.get_value(borrow=True).shape[0]
        self.test_set_size = self.test_in.get_value(borrow=True).shape[0]

        self.batch_size = 1
        self.train_data_available = True
        self.val_data_available = True
        self.test_data_available = True

    def train_input(self, batch_index):
        return self._get_subset(self.train_in, batch_index)

    def train_output(self, batch_index):
        return self._get_subset(self.train_out, batch_index)

    def val_input(self, batch_index):
        return self._get_subset(self.val_in, batch_index)

    def val_output(self, batch_index):
        return self._get_subset(self.val_out, batch_index)

    def test_input(self, batch_index):
        return self._get_subset(self.test_in, batch_index)

    def test_output(self, batch_index):
        return self._get_subset(self.test_out, batch_index)
