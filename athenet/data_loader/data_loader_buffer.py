"""Buffer for storing large network data."""

import numpy as np

import theano


class Buffer(object):
    """Buffer storing data from contiguous subsequence of minibatches.

    Content of a buffer is a 4-dimensional floating-point tensor.
    """
    def __init__(self):
        self.low = -1
        self.high = 0

        # Create a 4-dimensinal tensor shared variable for data. Exact size of
        # the tensor is determined when data is set, and can change over time.
        self._data = theano.shared(
            np.zeros((1, 1, 1, 1), dtype=theano.config.floatX),
            borrow=True)

    @property
    def data(self):
        """Shared variable representing data stored in a buffer."""
        return self._data

    def __getitem__(self, index):
        """Allow using buffer[index] instead of buffer.data[index]."""
        return self._data[index]

    def set(self, data, batch_index=None, n_of_batches=None):
        """Set buffer data.

        :data: Data to be stored in a buffer.
        :batch_index: Index of first minibatch that is contained in given
                      data.
        :n_of_batches: Number of minibatches that are contained in given data.
        """
        if batch_index:
            self.low = batch_index
            if n_of_batches:
                self.high = batch_index + n_of_batches
        self._data.set_value(
            np.asarray(np.concatenate(data, axis=0),
                       dtype=theano.config.floatX),
            borrow=True)
