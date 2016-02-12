"""Buffer for storing large network data."""

import numpy as np

import theano


class Buffer(object):
    """Buffer storing data from contiguous subsequence of minibatches.

    Content of a buffer is a 4-dimensional floating-point tensor.
    """
    def __init__(self, data_loader=None):
        """Create data Buffer.

        :data_loader: Instance of DataLoader that will be using Buffer.
        """
        self.begin = -1
        self.end = 0
        self.offset = theano.shared(0)
        self.parent = data_loader

        # Create a 4-dimensinal tensor shared variable for data. Exact size of
        # the tensor is determined when data is set, and can change over time.
        self._data = theano.shared(
            np.zeros((1, 1, 1, 1), dtype=theano.config.floatX),
            borrow=True)

    @property
    def data(self):
        """Shared variable representing data stored in a buffer."""
        return self._data

    def __getitem__(self, key):
        """Return minibatches of given indices.

        Return data is taken from data array, however key represents
        minibatch index, not direct index in data array. Effectively, buffer
        can be used as if it contained all of the minibatches data.

        Parent must be set before using this method, as minibatch size is
        needed to determine shift that has to be uses in data array.

        :key: Symbolic index or slice representing indices of minibatches to return.
        :return: Minibatches data.
        """
        shift = self.offset * self.parent.batch_size
        if isinstance(key, slice):
            start, stop, step = key.start, key.stop, key.step
            return self._data[start-shift:stop-shift:step]
        else:
            return self._data[key-shift]

    def set(self, data, batch_index=None, n_of_batches=None):
        """Set buffer data.

        :data: Data to be stored in a buffer.
        :batch_index: Index of first minibatch that is contained in given
                      data.
        :n_of_batches: Number of minibatches that are contained in given data.
        """
        if batch_index:
            self.begin = batch_index
            self.offset.set_value(batch_index)
            if n_of_batches:
                self.end = batch_index + n_of_batches
        self._data.set_value(
            np.asarray(np.concatenate(data, axis=0),
                       dtype=theano.config.floatX),
            borrow=True)

    def contains(self, batch_index):
        """Check if minibatch is contained in a buffer.

        :batch_index: Index of a minibatch.
        :return: True, if minibatch of a given index is contained in a buffer.
                 False otherwise.
        """
        return batch_index >= self.begin and batch_index < self.end
