"""ImageNet data loader."""

import os
import numpy as np
from scipy import misc
from collections import OrderedDict

import theano
import theano.tensor as T

from athenet.utils import get_bin_path, get_data_path
from athenet.data_loader import DataLoader


class ImageNetDataLoader(DataLoader):
    """ImageNet data loader."""

    name_prefix = 'ILSVRC'
    train_suffix = '_img_train'
    val_suffix = '_img_val'
    mean_rgb = [123, 117, 104]
    verbosity = 0

    def __init__(self, year, val_size=None, val_buffer_size=1):
        """Create ImageNet data loader.

        :year: Specifies which year's data should be loaded.
        :val_size: Maximal size of validation data. If None, then all
                   validation data will be used. Otherwise, val_size images
                   will be chosen randomly from the whole set.
        :val_buffer_size: Number of batches to be stored in memory.
        """
        super(ImageNetDataLoader, self).__init__()
        self._val_low = None
        self._val_high = None
        self._offset = theano.shared(0)
        self.val_buffer_size = val_buffer_size

        base_name = self.name_prefix + str(year)
        train_name = base_name + self.train_suffix
        val_name = base_name + self.val_suffix

        self.train_dir_name = train_name + '/'
        self.val_dir_name = val_name + '/'

        files = os.listdir(get_bin_path(self.val_dir_name))
        answers = OrderedDict()
        f = open(get_data_path(val_name + '.txt'), 'rb')
        while True:
            line = f.readline()
            if not line:
                break
            filename, answer = line.rsplit(' ', 1)
            answers[filename] = int(answer)
        f.close()

        self.val_files = np.asarray(answers.keys())
        val_answers = np.asarray(answers.values())
        self.val_set_size = len(self.val_files)

        if val_size and val_size < self.val_set_size:
            ind = np.random.permutation(self.val_set_size)[:val_size]
            self.val_files = self.val_files[ind]
            val_answers = val_answers[ind]
            self.val_set_size = val_size

        self.val_in = theano.shared(
            np.zeros((1, 3, 227, 227), dtype=theano.config.floatX),
            borrow=True)
        self.batch_size = 1

        self.val_out = theano.shared(val_answers, borrow=True)
        self.val_data_available = True

    def _get_img(self, filename):
        img = misc.imread(get_bin_path(filename))
        img = np.rollaxis(img, 2)
        img = img.reshape((1, 3, 227, 227))
        return np.asarray(img, dtype=float)

    def load_val_data(self, batch_index):
        if batch_index >= self._val_low and batch_index < self._val_high:
            return
        if self.verbosity > 0:
            print 'Load data'

        files = self._get_subset(self.val_files, batch_index,
                                 self.val_buffer_size)
        imgs = []
        for filename in files:
            img = self._get_img(self.val_dir_name + filename)
            r, g, b = np.split(img, 3, axis=1)
            r -= self.mean_rgb[0]
            g -= self.mean_rgb[1]
            b -= self.mean_rgb[2]
            img = np.concatenate([b, g, r], axis=1)
            img = img[:, :, ::-1, :]
            imgs += [img]

        imgs = np.concatenate(imgs, axis=0)
        self.val_in.set_value(
            np.asarray(imgs, dtype=theano.config.floatX), borrow=True)
        self._offset.set_value(batch_index)
        self._val_low = batch_index
        self._val_high = batch_index + self.val_buffer_size

    def val_input(self, batch_index):
        return self._get_subset(self.val_in, batch_index - self._offset)

    def val_output(self, batch_index):
        return self._get_subset(self.val_out, batch_index)
