"""Module for loading the MNIST data"""

import gzip
import cPickle
import numpy as np
import theano
import theano.tensor as T


def load_mnist_data(filename):
    """Load MNIST data from file.

    filename: Name of the file with MNIST data
    """
    f = gzip.open(filename, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    def shared_dataset(data_xy, borrow=True):
        """Create shared variables from given data.

        data_xy: data consisting of pairs (x, y)
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                            dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                            dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    return [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]


def download_mnist_data(filename):
    """Download MNIST data.

    filename: Name of the MNIST data file to be created
    """
    print 'Downloading MNIST data... ',
    mnist_origin = ('http://www.iro.umontreal.ca/~lisa/deep/data/mnist/'
                    'mnist.pkl.gz')
    urllib.urlretrieve(mnist_origin, filename)
    print 'Done.'
