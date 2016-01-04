"""ImageNet data loader."""

import os
import numpy as np
from scipy import misc
from collections import OrderedDict

import theano
import theano.tensor as T

from athenet.utils import DataLoader, get_bin_path, get_data_path

class ImageNetDataLoader(DataLoader):
    """ImageNet data loader."""

    name_prefix = 'ILSVRC'
    name_infix = '_img_'
    verbosity = 0

    def __init__(self, year, val_size=None, val_buffer_size=10):
        """Create ImageNet data loader.

        Only validation data are currently supported.

        year: Specifies which year's data should be loaded.
        val_size: Maximal size of validation data. If None, then all
                  validation data will be used.
        val_buffer_size: Number of batches to be stored in memory.
        """
        super(ImageNetDataLoader, self).__init__()
        self._val_low = None
        self._val_high = None

        self._offset = theano.shared(0)

        base_name = self.name_prefix + str(year) + self.name_infix
        self.val_name = base_name + 'val'
        self.val_dir_name = self.val_name + '/'
        self.val_buffer_size = val_buffer_size

        files = os.listdir(get_bin_path(self.val_dir_name))
        answers = OrderedDict()
        f = open(get_data_path(self.val_name + '.txt'), 'rb')
        while True:
            line = f.readline()
            if not line:
                break
            filename, answer = line.rsplit(' ', 1)
            if filename in files:
                answers[filename] = int(answer)
        f.close()

        self.val_files = answers.keys()
        val_answers = answers.values()
        self.val_set_size = len(self.val_files)

        if val_size and val_size < self.val_set_size:
            self.val_files = self.val_files[:val_size]
            val_answers = val_answers[:val_size]
            self.val_set_size = val_size

        self.val_in = theano.shared(
            np.zeros((1, 3, 227, 227),
                     dtype=theano.config.floatX),
            borrow=True)
        self.batch_size = 1

        self.val_out = theano.shared(
            np.asarray(val_answers, dtype='int32'), borrow=True)
        self.val_data_available = True

    def _get_img(self, filename):
        img = misc.imread(get_bin_path(filename))
        img = np.rollaxis(img, 2)
        return img.reshape((1, 3, 227, 227))

    def load_val_data(self, batch_index):
        if (batch_index >= self._val_low) and (batch_index < self._val_high):
            return
        if self.verbosity > 0:
            print 'Load data'

        files = self._get_subset(self.val_files, batch_index,
                                 self.val_buffer_size)
        val_input = self._get_img(self.val_dir_name + files[0])
        for filename in files[1:]:
            img = self._get_img(self.val_dir_name + filename)
            val_input = np.concatenate([val_input, img], axis=0)

        self.val_in.set_value(
            np.asarray(val_input, dtype=theano.config.floatX), borrow=True)
        self._offset.set_value(batch_index)
        self._val_low = batch_index
        self._val_high = batch_index + self.val_buffer_size

    def val_input(self, batch_index):
        return self._get_subset(self.val_in, batch_index - self._offset)

    def val_output(self, batch_index):
        return self._get_subset(self.val_out, batch_index)
