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

    def __init__(self, year, val_size=None):
        """Create ImageNet data loader.

        Only validation data are currently supported.

        year: Specifies which year's data should be loaded.
        val_size: Maximal size of validation data.
        """
        super(ImageNetDataLoader, self).__init__()

        base_name = self.name_prefix + str(year) + self.name_infix
        self.val_name = base_name + 'val'
        self.val_dir_name = self.val_name + '/'
        self.val_in = theano.shared(
            np.zeros((1, 3, 227, 227), dtype=theano.config.floatX),
            borrow=True)

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

        self.val_out = theano.shared(
            np.asarray(val_answers, dtype='int32'), borrow=True)
        self.val_data_available = True

        self.batch_size = 1

    def _get_img(self, filename):
        img = misc.imread(get_bin_path(filename))
        h, w = img.shape[0:2]
        if len(img.shape) == 2:
            shape = (h, w, 1)
            img = np.reshape(img, shape)
            img = np.tile(img, (1, 1, 3))
        m = min(h, w)
        img = img[(h-m)/2:(h+m)/2, (w-m)/2:(w+m)/2, :]
        img = misc.imresize(img, size=(227, 227))
        img = np.swapaxes(img, 1, 2)
        img = np.swapaxes(img, 0, 1)
        img = np.asarray(img, dtype=float)
        return img.reshape((1, 3, 227, 227))

    def load_val_data(self, batch_index):
        files = self._get_subset(self.val_files, batch_index)

        val_input = self._get_img(self.val_dir_name + files[0])
        for filename in files[1:]:
            img = self._get_img(self.val_dir_name + filename)
            val_input = np.concatenate([val_input, img], axis=0)

        self.val_in.set_value(
            np.asarray(val_input, dtype=theano.config.floatX), borrow=True)

    def val_input(self, batch_index):
        return self.val_in

    def val_output(self, batch_index):
        return self._get_subset(self.val_out, batch_index)
