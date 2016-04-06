"""Auxiliary functions."""

import cPickle as pickle
import os
import sys
import gzip
import urllib
import numpy

import theano

from athenet.utils import BIN_DIR, DATA_DIR


def load_data_from_pickle(filename):
    """Load data from pickle file.

    :param filename: File with pickled data, may be gzipped.
    :return: Data loaded from file.
    """
    try:
        f = gzip.open(filename, 'rb')
        data = pickle.load(f)
    except:
        f = open(filename, 'rb')
        data = pickle.load(f)
    f.close()
    return data


def save_data_to_pickle(data, filename):
    """Saves data to gzipped pickle file.

    :param data: Data to be saved.
    :param filename: Name of file to save data.
    """
    with gzip.open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_data(filename, url=None):
    """Load data from file, download file if it doesn't exist.

    :param filename: File with pickled data, may be gzipped.
    :param url: Url for downloading file.
    :return: Unpickled data.
    """
    if not os.path.isfile(filename):
        if not url:
            return None
        download_file(filename, url)

    data = load_data_from_pickle(filename)
    return data


def download_file(filename, url):
    """Download file from given url.

    :param filename: Name of a file to be downloaded.
    :param url: Url for downloading file.
    """
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    print 'Downloading ' + os.path.basename(filename) + '...',
    sys.stdout.flush()
    urllib.urlretrieve(url, filename)
    print 'Done'


def get_data_path(name):
    """Return absolute path to the data file.

    :param name: Name of the file.
    :return: Full path to the file.
    """
    return os.path.join(DATA_DIR, name)


def get_bin_path(name):
    """Return absolute path to the binary data file.

    :param name: Name of the file.
    :return: Full path to the file.
    """
    return os.path.join(BIN_DIR, name)


def zero_fraction(network):
    """Returns fraction of zeros in weights of Network.

    Biases are not considered.

    :param network: Network for which we count fraction of zeros.
    :return: Fraction of zeros.
    """
    params = [layer.W for layer in network.weighted_layers]
    n_non_zero = 0
    n_fields = 0
    for param in params:
        n_fields += numpy.size(param)
        n_non_zero += numpy.count_nonzero(param)
    n_zero = n_fields - n_non_zero
    return (1.0 * n_zero) / (1.0 * n_fields)


def count_zeros_in_layer(layer):
    all_weights = layer.W.size
    return all_weights - numpy.count_nonzero(layer.W), all_weights


def count_zeros(network):
    """Returns zeros in weights of Network.

    Biases are not considered.

    :param network: Network for which we count zeros.
    :return: List of tuples (number of weights being zero, number of weights)
        for each layer.
    """
    return [count_zeros_in_layer(layer) for layer in network.weighted_layers]

len_prev = 0


def overwrite(text='', length=None):
    """Write text in a current line, overwriting previously written text.

    Previously written text also needs to be written using this function for
    it to work properly. Otherwise optional argument length can be given to
    specify length of a previous text.

    :param text: Text to be written.
    :param length: Length of a previous text.
    """
    global len_prev
    if length is None:
        length = len_prev
    print '\r' + ' '*length,
    print '\r' + text,
    len_prev = len(text)


def cudnn_available():
    """Check if cuDNN is available.

    :return: True, if cuDNN is available, False otherwise.
    """
    try:
        return theano.sandbox.cuda.dnn_available()
    except:
        return False
