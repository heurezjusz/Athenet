"""Module for loading the MNIST data"""

import gzip
import cPickle as pickle
import urllib
import os
import sys
import numpy as np

import theano
import theano.tensor as T


_MNIST_FILENAME = os.path.join(os.path.dirname(__file__),
                               '../../bin/mnist.pkl.gz')
_MNIST_ORIGIN = ('http://www.iro.umontreal.ca/~lisa/deep/data/mnist/'
                 'mnist.pkl.gz')


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


def load_mnist_data(filename=_MNIST_FILENAME, url=_MNIST_ORIGIN):
    """Load MNIST data from file.

    filename: Name of the file with MNIST data.
    url: Url for downloading MNIST data.
    return: List of training, validation and test data in the format (x, y).
    """
    train_set, valid_set, test_set = load_data(filename, url)

    test_set_x, test_set_y = _mnist_shared_dataset(test_set)
    valid_set_x, valid_set_y = _mnist_shared_dataset(valid_set)
    train_set_x, train_set_y = _mnist_shared_dataset(train_set)

    return [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]


def _mnist_shared_dataset(data, borrow=True):
        """Create shared variables from given data.

        data: Data consisting of pairs (x, y).
        return: Theano shared variables created from data.
        """
        data_x, data_y = data
        data_x = np.resize(data_x, (data_x.shape[0], 28, 28, 1))
        shared_x = theano.shared(np.asarray(data_x,
                                            dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                            dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')
