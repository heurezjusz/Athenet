"""Data loader class and functions."""

import gzip
import cPickle as pickle
import urllib
import os
import sys


def load_data(filename, url=None):
    """Load data from file, download file if it doesn't exist.

    filename: File with pickled data.
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

    f = gzip.open(filename, 'rb')
    data = pickle.load(f)
    f.close()
    return data


class DataLoader(object):
    """Data loader."""
    def __init__(self):
        self.batch_size = 1

    def _get_subset(self, data, batch_index):
        return data[batch_index*self.batch_size:
                    (batch_index+1)*self.batch_size]

    def train_input(self, batch_index):
        """Return minibatch of training data input.

        batch_index: Batch index.
        return: List of minibatch training data input.
        """
        raise NotImplementedError()

    def train_output(self, batch_index):
        """Return minibatch of training data output.

        batch_index: Batch index.
        return: List of minibatch training data output.
        """
        raise NotImplementedError()

    def valid_input(self, batch_index):
        """Return minibatch of validation data input.

        batch_index: Batch index.
        return: List of minibatch validation data input.
        """
        raise NotImplementedError()

    def valid_output(self, batch_index):
        """Return minibatch of validation data output.

        batch_index: Batch index.
        return: List of minibatch validation data output.
        """
        raise NotImplementedError()

    def test_input(self, batch_index):
        """Return minibatch of testing data input.

        batch_index: Batch index.
        return: List of minibatch testing data input.
        """
        raise NotImplementedError()

    def test_output(self, batch_index):
        """Return minibatch of testing data output.

        batch_index: Batch index.
        return: List of minibatch testing data output.
        """
        raise NotImplementedError()
