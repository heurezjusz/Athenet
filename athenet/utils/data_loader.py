"""Data loader class and functions."""

import gzip
import cPickle as pickle
import urllib
import os
import sys


def load_data(filename, url=None):
    """Load data from file, download file if it doesn't exist.

    filename: File with pickled data, may be gzipped.
    url: Url for downloading file.
    return: Unpickled data.
    """
    if not os.path.isfile(filename):
        if not url:
            return None
        else:
            directory = os.path.dirname(filename)
            if not os.path.exists(directory):
                os.makedirs(directory)

            print 'Downloading ' + os.path.basename(filename) + '...',
            sys.stdout.flush()
            urllib.urlretrieve(url, filename)
            print 'Done'

    try:
        f = gzip.open(filename, 'rb')
        data = pickle.load(f)
    except:
        f = open(filename, 'rb')
        data = pickle.load(f)
    f.close()
    return data


class DataLoader(object):
    """Data loader."""
    def __init__(self):
        self._batch_size = None
        self.n_train_batches = None
        self.n_valid_batches = None
        self.n_test_batches = None

        self.batch_size = 1

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value

        self.n_train_batches = self.train_set_size / self.batch_size
        self.n_valid_batches = self.valid_set_size / self.batch_size
        self.n_test_batches = self.test_set_size / self.batch_size

    def _get_subset(self, data, batch_index):
        return data[batch_index*self.batch_size:
                    (batch_index+1)*self.batch_size]

    def train_input(self, batch_index):
        """Return minibatch of training data input.

        batch_index: Batch index.
        return: Minibatch of training data input.
        """
        raise NotImplementedError()

    def train_output(self, batch_index):
        """Return minibatch of training data output.

        batch_index: Batch index.
        return: Minibatch of training data output.
        """
        raise NotImplementedError()

    def valid_input(self, batch_index):
        """Return minibatch of validation data input.

        batch_index: Batch index.
        return: Minibatch of validation data input.
        """
        raise NotImplementedError()

    def valid_output(self, batch_index):
        """Return minibatch of validation data output.

        batch_index: Batch index.
        return: Minibatch of validation data output.
        """
        raise NotImplementedError()

    def test_input(self, batch_index):
        """Return minibatch of testing data input.

        batch_index: Batch index.
        return: Minibatch of testing data input.
        """
        raise NotImplementedError()

    def test_output(self, batch_index):
        """Return minibatch of testing data output.

        batch_index: Batch index.
        return: Minibatch of testing data output.
        """
        raise NotImplementedError()
