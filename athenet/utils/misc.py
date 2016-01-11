"""Auxiliary functions."""

import cPickle as pickle
import os
import sys
import gzip
import urllib

from athenet.utils import BIN_DIR, DATA_DIR


def load_data(filename, url=None):
    """Load data from file, download file if it doesn't exist.

    :filename: File with pickled data, may be gzipped.
    :url: Url for downloading file.
    :return: Unpickled data.
    """
    if not os.path.isfile(filename):
        if not url:
            return None
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
        download_file(filename, url)

    try:
        f = gzip.open(filename, 'rb')
        data = pickle.load(f)
    except:
        f = open(filename, 'rb')
        data = pickle.load(f)
    f.close()
    return data


def download_file(filename, url):
    """Download file from given url.

    :filename: Name of a file to be downloaded.
    :url: Url for downloading file.
    """
    print 'Downloading ' + os.path.basename(filename) + '...',
    sys.stdout.flush()
    urllib.urlretrieve(url, filename)
    print 'Done'


def get_data_path(name):
    """Return absolute path to the data file.

    :name: Name of the file.
    :return: Full path to the file.
    """
    return os.path.join(DATA_DIR, name)


def get_bin_path(name):
    """Return absolute path to the binary data file.

    :name: Name of the file.
    :return: Full path to the file.
    """
    return os.path.join(BIN_DIR, name)
