"""Auxiliary functions."""

import cPickle as pickle
import os
import sys
import gzip
import urllib


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
