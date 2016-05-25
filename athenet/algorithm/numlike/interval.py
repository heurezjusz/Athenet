"""Intervals including special functions for
sparsifying.

This module contains Interval class and auxiliary objects.
"""

from numlike import Numlike


class Interval(Numlike):
    """Interval matrix class

    Represents matrix of intervals. Behaves like limited numpy.ndarray of
    intervals.

    Operation combining many diffrent subclasses of Interval are not supported

    """

    NEUTRAL_LOWER = 0.0
    NEUTRAL_UPPER = 0.0
    NEUTRAL_VALUES = (NEUTRAL_LOWER, NEUTRAL_UPPER)

    DEFAULT_LOWER = 0.0
    DEFAULT_UPPER = 255.0
    DEFAULT_VALUES = (DEFAULT_LOWER, DEFAULT_UPPER)

    def __init__(self, lower, upper):
        """Creates interval.

        :param lower: lower bound of Interval to be set
        :param upper: upper bound of Interval to be set

        .. note:: lower must be lower than upper. It is not being checked.
        """
        super(Interval, self).__init__()
        self.lower = lower
        self.upper = upper

    @staticmethod
    def construct(lower, upper):
        raise NotImplementedError

    def __getitem__(self, at):
        """Returns specified slice of interval as a interval.

        :param at: coordinates / slice to be taken
        :rtype: Interval

        .. note:: Does not copy data.
        """
        return self.construct(self.lower[at], self.upper[at])

    @property
    def shape(self):
        """Returns shape of interval. Checks only 'lower' matrix.

        .. note:: does not require self.upper for computations. Therefore it is
        not safe, but faster.
        """
        return self.lower.shape

    def __add__(self, other):
        """Returns sum of two intervals.

        :param other: matrix to be added
        :type other: Interval or numpy.ndarray or float
        :rtype: Interval
        """
        if isinstance(other, Interval):
            res_lower = self.lower + other.lower
            res_upper = self.upper + other.upper
        else:
            res_lower = self.lower + other
            res_upper = self.upper + other
        return self.construct(res_lower, res_upper)

    def __sub__(self, other):
        """Returns difference between two intervals.

        :param other: matrix to be subtracted
        :type other: Interval or numpy.ndarray or float
        :rtype: Interval
        """
        if isinstance(other, Interval):
            res_lower = self.lower - other.upper
            res_upper = self.upper - other.lower
        else:
            res_lower = self.lower - other
            res_upper = self.upper - other
        return self.construct(res_lower, res_upper)

    def __rdiv__(self, other):
        """Returns quotient of other and self.

        :param other: dividend
        :type other: Interval or numpy.ndarray or float
        :rtype: Interval

        .. warning:: Divisor (self) should not contain zero.
        """
        if isinstance(other, Interval):
            # Should never happen. __div__ should be used instead.
            raise NotImplementedError
        else:
            lower = self.lower
            upper = self.upper
            l_o = (other > 0)
            if l_o:
                return self.construct(other / upper, other / lower)
            else:
                return self.construct(other / lower, other / upper)

    def amax(self, axis=None, keepdims=False):
        """Returns maximum of an Interval along an axis.

        Works like theano.tensor.max.
        :param axis: axis or axes along which to compute the maximum
        :param keepdims: If this is set to True, the axes which are reduced are
                         left in the result as dimensions with size one. With
                         this option, the result will broadcast correctly
                         against the original tensor.
        :type keepdims: boolean
        """
        lower = self.lower.max(axis=axis, keepdims=keepdims)
        upper = self.upper.max(axis=axis, keepdims=keepdims)
        return self.construct(lower, upper)

    def flatten(self):
        """Flattens Interval

        :rtype: Interval
        """
        return self.construct(self.lower.flatten(), self.upper.flatten())

    def reshape(self, shape):
        """Reshapes interval

        :param shape: tuple of integers
        :rtype: Interval
        """
        return self.construct(self.lower.reshape(shape),
                              self.upper.reshape(shape))

    def sum(self, axis=None, dtype=None, keepdims=False):
        """Sum of array elements over a given axis like in numpy.ndarray.

        :param integer or None axis: axis along which this function sums
        :param type or None dtype: just like dtype argument in
                                   theano.tensor.sum
        :param Boolean keepdims: Whether to keep squashed dimensions of size 1
        """
        return self.construct(
            self.lower.sum(axis=axis, dtype=dtype, keepdims=keepdims),
            self.upper.sum(axis=axis, dtype=dtype, keepdims=keepdims)
        )

    @property
    def T(self):
        """Tensor transposition like in numpy.ndarray.

        :rtype: Interval
        """
        return self.construct(self.lower.T, self.upper.T)

    @staticmethod
    def _reshape_for_padding(layer_input, image_shape, batch_size, padding,
                             value=0.0):
        raise NotImplementedError

    def reshape_for_padding(self, shape, padding, lower_val=None,
                            upper_val=None):
        """Returns padded Interval.

        :param tuple of 4 integers shape: shape of input in format
                                          (batch size, number of channels,
                                           height, width)
        :param pair of integers padding: padding to be applied
        :param float lower_val: value of lower bound in new fields
        :param float upper_val: value of upper bound in new fields
        :returns: padded layer_input
        :rtype: Interval
        """
        if lower_val is None:
            lower_val = self.NEUTRAL_LOWER
        if upper_val is None:
            upper_val = self.NEUTRAL_UPPER
        n_batches, n_in, h, w = shape
        padded_low = self._reshape_for_padding(self.lower, (h, w, n_in),
                                               n_batches, padding, lower_val)
        padded_upp = self._reshape_for_padding(self.upper, (h, w, n_in),
                                               n_batches, padding, upper_val)
        return self.construct(padded_low, padded_upp)

    def __repr__(self):
        """Standard repr method."""
        return 'vvvvv\n' + repr(self.lower) + '\n=====\n' + repr(self.upper) \
               + '\n^^^^^'

    def __str__(self):
        """"Standard str method."""
        return 'vvvvv\n' + str(self.lower) + '\n=====\n' + str(self.upper) \
               + '\n^^^^^'
