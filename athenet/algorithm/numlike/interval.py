"""Intervals implemented in Theano including special functions for
sparsifying.

This module contains Interval class and auxiliary objects.
"""

from numlike import Numlike
from theano import function
from theano import tensor as T
from theano import shared

import numpy

NEUTRAL_INTERVAL_LOWER = 0.0
NEUTRAL_INTERVAL_UPPER = 0.0
NEUTRAL_INTERVAL_VALUES = (NEUTRAL_INTERVAL_LOWER, NEUTRAL_INTERVAL_UPPER)

DEFAULT_INTERVAL_LOWER = 0.0
DEFAULT_INTERVAL_UPPER = 255.0
DEFAULT_INTERVAL_VALUES = (DEFAULT_INTERVAL_LOWER, DEFAULT_INTERVAL_UPPER)


class Interval(Numlike):
    """Theano interval matrix class

    Represents matrix of intervals. Behaves like limited numpy.ndarray of
    intervals.

    .. note:: Should be treated as interval type with bounds as Theano nodes.
              Operations on Interval create nodes in Theano graph. In order to
              read result of given operations, use eval method.
    """

    def __init__(self, lower, upper):
        """Creates interval.

        :param theano tensor lower: lower bound of Interval to be set
        :param theano tensor upper: upper bound of Interval to be set

        .. note:: lower must be lower than upper. It is not being checked.
        """
        super(Interval, self).__init__()
        self.lower = lower
        self.upper = upper

    def __getitem__(self, at):
        """Returns specified slice of interval as a interval.

        :param at: coordinates / slice to be taken
        :rtype: Interval

        .. note:: Does not copy data.
        """
        return Interval(self.lower[at], self.upper[at])

    def __setitem__(self, at, other):
        """Just like Theano set_subtensor function, but as a operator.

        :param at: coordinates / slice to be set
        :param other: data to be put at 'at'
        :type other: Interval
        """
        self.lower = T.set_subtensor(self.lower[at], other.lower)
        self.upper = T.set_subtensor(self.upper[at], other.upper)

    @property
    def shape(self):
        """Returns shape of interval. Checks only 'lower' matrix.

        :rtype: theano tuple of integers

        .. note:: does not require self.upper for computations. Therefore it is
        not safe, but faster.
        """
        return self.lower.shape

    def __add__(self, other):
        """Returns sum of two intervals.

        :param other: matrix to be added
        :type other: Interval or numpy.ndarray
        :rtype: Interval
        """
        if isinstance(other, Interval):
            res_lower = self.lower + other.lower
            res_upper = self.upper + other.upper
        else:
            res_lower = self.lower + other
            res_upper = self.upper + other
        return Interval(res_lower, res_upper)

    def __sub__(self, other):
        """Returns difference between two intervals.

        :param other: matrix to be subtracted
        :type other: Interval or numpy.ndarray
        :rtype: Interval
        """
        if isinstance(other, Interval):
            res_lower = self.lower - other.upper
            res_upper = self.upper - other.lower
        else:
            res_lower = self.lower - other
            res_upper = self.upper - other
        return Interval(res_lower, res_upper)

    def __mul__(self, other):
        """Returns product of two intervals.

        :param other: matrix to be multiplied
        :type other: Interval or numpy.ndarray
        :rtype: Interval
        """
        if isinstance(other, Interval):
            ll = self.lower * other.lower
            lu = self.lower * other.upper
            ul = self.upper * other.lower
            uu = self.upper * other.upper
            l = T.minimum(ll, lu)
            l = T.minimum(l, ul)
            l = T.minimum(l, uu)
            u = T.maximum(ll, lu)
            u = T.maximum(u, ul)
            u = T.maximum(u, uu)
            return Interval(l, u)
        else:
            ll = self.lower * other
            uu = self.upper * other
            l = T.minimum(ll, uu)
            u = T.maximum(ll, uu)
            return Interval(l, u)

    def __div__(self, other):
        """Returns quotient of self and other.

        :param other: divisor
        :type other: Interval or numpy.ndarray or float
        :rtype: Interval

        .. warning:: Divisor should not contain zero.
        """
        lower = self.lower
        upper = self.upper
        if isinstance(other, Interval):
            o_lower = other.lower
            o_upper = other.upper
            a = T.switch(T.gt(o_lower, 0.0), 1, 0)  # not(b_la), b_ua
            b = T.switch(T.gt(lower, 0.0), 1, 0)
            c = T.switch(T.gt(upper, 0.0), 1, 0)
            b_lb = T.or_(T.and_(a, b),
                         T.and_(1 - a, c))
            b_ub = T.or_(1 - T.or_(a, b),
                         T.and_(a, 1 - c))
            la = T.switch(a, lower, upper)
            ua = T.switch(a, upper, lower)
            lb = T.switch(b_lb, o_upper, o_lower)
            ub = T.switch(b_ub, o_upper, o_lower)
            return Interval(la / lb, ua / ub)
        else:
            if other > 0:
                return Interval(lower / other, upper / other)
            else:
                return Interval(upper / other, lower / other)

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
                return Interval(other / upper, other / lower)
            else:
                return Interval(other / lower, other / upper)

    def reciprocal(self):
        """Returns reciprocal of the interval.

        It is a partial reciprocal function. Does not allow 0 to be within
        interval. Should not be treated as general reciprocal function.

        :rtype: Interval
        """
        # Note: Could be considered whether not to use input check.
        # If 0 is within interval, returns 1/0 that, we hope, will throw
        # some exception on the device. Be careful with this.

        # Input check below causes program interrupt if any _has_zero happened.
        # return Interval(switch(self._has_zero(),
        #                       T.constant(1)/T.constant(0),
        #                       T.inv(self.upper)),
        #                T.inv(self.lower))
        return Interval(T.inv(self.upper), T.inv(self.lower))

    def neg(self):
        """For interval [a, b], returns interval [-b, -a]."""
        return Interval(T.neg(self.upper), T.neg(self.lower))

    def exp(self):
        """Returns interval representing the exponential of the interval."""
        return Interval(T.exp(self.lower), T.exp(self.upper))

    def square(self):
        """For interval I, returns I' such that for any x in I, I' contains
        x*x and no other.
        :rtype: Interval

        :Example:

        >>> from athenet.algorithm.numlike import Interval
        >>> import numpy
        >>> a = numpy.array([-1])
        >>> b = numpy.array([1])
        >>> i = Interval(a, b)
        >>> s = i.square()
        >>> s.eval()
        (array([0]), array([1]))
        """
        lsq = self.lower * self.lower
        usq = self.upper * self.upper
        u = T.maximum(lsq, usq)
        l = T.switch(self._has_zero(), 0, T.minimum(lsq, usq))
        return Interval(l, u)

    def power(self, exponent):
        """For interval i, returns i^exponent.

        :param exponent: Number to be passed as exponent to i^exponent.
        :type exponent: integer or float
        :rtype: Interval

        .. note:: If interval contains some elements lower/equal to 0, exponent
                  should be integer."""
        # If You want to understand what is happening here, make plot of
        # f(x, y) = x^y domain. 'if's divide this domain with respect to
        # monotonicity.
        le = T.pow(self.lower, exponent)
        ue = T.pow(self.upper, exponent)
        if isinstance(exponent, (int, long)):
            if exponent > 0:
                if exponent % 2 == 0:
                    l = T.switch(self._has_zero(), 0, T.minimum(le, ue))
                    u = T.maximum(le, ue)
                else:
                    l = le
                    u = ue
            else:
                if exponent % 2 == 0:
                    l = T.minimum(le, ue)
                    u = T.maximum(le, ue)
                else:
                    l = ue
                    u = le
        else:
            # Assumes self.lower >= 0. Otherwise it is incorrectly defined.
            # There is no check.
            if exponent > 0:
                l = le
                u = ue
            else:
                l = ue
                u = le
        return Interval(l, u)

    def dot(self, other):
        """Returns dot product of Interval(self) vector and a number array
        (other).

        :param numpy.ndarray or theano.tensor other: number array to be
                                                     multiplied
        :rtype: Interval
        """
        lower = self.lower
        upper = self.upper
        other_negative = T.minimum(other, 0.0)
        other_positive = T.maximum(other, 0.0)
        lower_pos_dot = T.dot(lower, other_positive)
        lower_neg_dot = T.dot(lower, other_negative)
        upper_pos_dot = T.dot(upper, other_positive)
        upper_neg_dot = T.dot(upper, other_negative)
        res_lower = lower_pos_dot + upper_neg_dot
        res_upper = upper_pos_dot + lower_neg_dot
        return Interval(res_lower, res_upper)

    def max(self, other):
        """Returns interval such that for any numbers (x, y) in a pair of
        corresponding intervals in (self, other) arrays, max(x, y) is in result
        and no other.

        :param other: Interval to be compared
        :type other: Interval or theano.tensor
        :rtype: Interval
        """
        if isinstance(other, Interval):
            return Interval(T.maximum(self.lower, other.lower),
                            T.maximum(self.upper, other.upper))
        else:
            return Interval(T.maximum(self.lower, other),
                            T.maximum(self.upper, other))

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
        return Interval(lower, upper)

    def reshape(self, shape):
        """Reshapes interval tensor like theano Tensor.

        :param shape: Something that can be converted to a symbolic vector of
                      integers.
        """
        return Interval(self.lower.reshape(shape),
                        self.upper.reshape(shape))

    def flatten(self):
        """Flattens interval tensor like theano Tensor.

        :return: Variable with same dtype as x and outdim dimensions.
        :rtype: Variable with the same shape as x in the leading outdim-1
                dimensions, but with all remaining dimensions of x collapsed
                into the last dimension.
        """
        return Interval(self.lower.flatten(),
                        self.upper.flatten())

    def sum(self, axis=None, dtype=None, keepdims=False):
        """Tensor sum operation like in numpy.ndarray.

        :param integer or None axis: axis along which this function sums
        :param type or None dtype: just like dtype argument in
                                   theano.tensor.sum
        :param Boolean keepdims: Whether to keep squashed dimensions of size 1
        """
        return Interval(self.lower.sum(axis=axis, dtype=dtype,
                                       keepdims=keepdims),
                        self.upper.sum(axis=axis, dtype=dtype,
                                       keepdims=keepdims))

    def abs(self):
        """Returns absolute value of Interval."""
        lower = T.switch(T.gt(self.lower, 0.0), self.lower,
                         T.switch(T.lt(self.upper, 0.0), -self.upper, 0.0))
        upper = T.maximum(-self.lower, self.upper)
        return Interval(lower, upper)

    @property
    def T(self):
        """Tensor transposition like in numpy.ndarray."""
        return Interval(self.lower.T,
                        self.upper.T)

    @staticmethod
    def from_shape(shp, neutral=True, lower_val=None,
                   upper_val=None):
        """Returns Interval of shape shp with given lower and upper values.

        :param tuple of integers or integer shp : shape of created Interval
        :param Boolean neutral: if True sets (lower_val, upper_val) to
                                NEUTRAL_INTERVAL_VALUES, otherwise to
                                DEFAULT_INTERVAL_VALUES, works only if pair is
                                not set by passing arguments.
        :param float lower_val: value of lower bound
        :param float upper_val: value of upper bound
        """
        if lower_val > upper_val:
            raise ValueError("lower_val > upper_val in newly created Interval")
        if lower_val is None:
            lower_val = NEUTRAL_INTERVAL_LOWER if neutral else \
                        DEFAULT_INTERVAL_LOWER
        if upper_val is None:
            upper_val = NEUTRAL_INTERVAL_UPPER if neutral else \
                        DEFAULT_INTERVAL_UPPER
        lower_array = numpy.ndarray(shp)
        upper_array = numpy.ndarray(shp)
        lower_array.fill(lower_val)
        upper_array.fill(upper_val)
        lower = shared(lower_array)
        upper = shared(upper_array)
        return Interval(lower, upper)

    def eval(self, eval_map=None):
        """Evaluates interval in terms of theano TensorType eval method.

        :param eval_map: map of Theano variables to be set, just like in
                          theano.tensor.dtensorX.eval method
        :type eval_map: {theano.tensor: numpy.ndarray} dict

        :rtype: (lower, upper) pair of numpy.ndarrays
        """
        has_args = eval_map is not None
        if has_args:
            has_args = len(eval_map) != 0
        if not has_args:
            try:
                f = function([], [self.lower, self.upper])
                rlower, rupper = f()
                return rlower, rupper
            except:
                return self.lower, self.upper
        keys = eval_map.keys()
        values = eval_map.values()
        f = function(keys, [self.lower, self.upper])
        rlower, rupper = f(*values)
        return rlower, rupper

    def __repr__(self):
        """Standard repr method."""
        return '[' + repr(self.lower) + ', ' + repr(self.upper) + ']'

    def __str__(self):
        """"Standard str method."""
        return '[' + str(self.lower) + ', ' + str(self.upper) + ']'

    def _has_zero(self):
        """For any interval in Interval, returns whether is contains zero.

        :rtype: Boolean
        """
        return T.and_(T.lt(self.lower, 0.0), T.gt(self.upper, 0.0))
