"""Intervals in Theano including special functions for sparsifying."""

from athenet.sparsifying.utils.numlike import Numlike
from theano import function
from theano import tensor as T
from theano.ifelse import ifelse


class Interval(Numlike):
    """Theano interval matrix class

    represents matrix of interval. Behaves like limited numpy.ndarray.

    .. note:: Should be treated as interval type with bounds as Theano nodes.
              Opetations on Interval create nodes in Theano graph. In order to
              read result of given operations, use eval method.
    """

    def __init__(self, lower, upper):
        """Creates interval.

        :lower: lower bound of Interval to be set
        :upper: upper bound of Interval to be set

        .. note:: lower must be lower than upper. It is not being checked.
        """
        self.lower = lower
        self.upper = upper

    def __getitem__(self, at):
        """Returns specified slice of interval as a interval.

        :at: Coordinates / slice to be taken.

        Does not copy data.
        """
        return Interval(self.lower[at], self.upper[at])

    def __setitem__(self, at, other):
        """Just like Theano set_subtensor function, but as a operator.

        :at: Coordinates / slice to be set.
        :other: Data to be put at 'at'

        """
        self.lower = T.set_subtensor(self.lower[at], other.lower)
        self.upper = T.set_subtensor(self.upper[at], other.upper)

    def __repr__(self):
        """Standard repr method."""
        return '[' + repr(self.lower) + ', ' + repr(self.upper) + ']'

    def __str__(self):
        """"Standard str method."""
        return '[' + str(self.lower) + ', ' + str(self.upper) + ']'

    def shape(self):
        """Returns shape of interval. Checks only 'lower' matrix.

        Note: does not require self.upper for computations. Therefore it is not
        safe, but faster."""
        return self.lower.shape

    def __add__(self, other):
        """Returns sum of two intervals.

        :other: Interval or numpy.ndarray to be added."""
        if isinstance(other, Interval):
            res_lower = self.lower + other.lower
            res_upper = self.upper + other.upper
        else:
            res_lower = self.lower + other
            res_upper = self.upper + other
        return Interval(res_lower, res_upper)

    __radd__ = __add__

    def __sub__(self, other):
        """Returns difference between two intervals.

        :other: Interval or numpy.ndarray to be subtracted."""
        if isinstance(other, Interval):
            res_lower = self.lower - other.upper
            res_upper = self.upper - other.lower
        else:
            res_lower = self.lower - other
            res_upper = self.upper - other
        return Interval(res_lower, res_upper)

    def __rsub__(self, other):
        """Returns diffeerence between number and interval.

        :other: A number that self will be subtracted from."""
        res_lower = other - self.upper
        res_upper = other - self.lower
        return Interval(res_lower, res_upper)

    def __mul__(self, other):
        """Returns product of two intervals.

        :other: Interval or numpy.ndarray to be multiplied."""
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

    __rmul__ = __mul__

    def __div__(self, other):
        """Returns quotient of self and other.

        :param other: Divisor.
        :type other: Interval or numpy.ndarray

        .. warning:: Divisor should not contain zero."""
        lower = self.lower
        upper = self.upper
        if isinstance(other, Interval):
            o_lower = other.lower
            o_upper = other.upper
            A = T.switch(T.gt(o_lower, 0.0), 1, 0)  # not(b_la), b_ua
            B = T.switch(T.gt(lower, 0.0), 1, 0)
            C = T.switch(T.gt(upper, 0.0), 1, 0)
            b_lb = T.or_(T.and_(A, B),
                         T.and_(1 - A, C))
            b_ub = T.or_(1 - T.or_(A, B),
                         T.and_(A, 1 - C))
            la = T.switch(A, lower, upper)
            ua = T.switch(A, upper, lower)
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

        :param other: Dividend.
        :type other: Interval or numpy.ndarray

        .. warning:: Divisor (self) should not contain zero."""
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

    def _has_zero(self):
        """For any interval in Interval, returns whether is contains zero."""
        return T.and_(T.lt(self.lower, 0), T.gt(self.upper, 0))

    def reciprocal(self):
        """Returns reciprocal of the interval.

        It is a partial reciprocal function. Does not allow 0 to be within
        interval. Should not be treated as general reciprocal function."""
        # Note: Could be concidered whether not to use input check.
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

        e.g [-1, 1].square() == [0, 1]"""
        lsq = self.lower * self.lower
        usq = self.upper * self.upper
        u = T.maximum(lsq, usq)
        l = T.switch(self._has_zero(), 0, T.minimum(lsq, usq))
        return Interval(l, u)

    def power(self, exponent):
        """For interval i, returns i^exponent.

        :exponent: Number to be passed as exponent to i^exponent.

        Note: If interval contains some elements lower/equal to 0, exponent
        should be integer."""
        # If You want to understand what is happening here, make plot of
        # f(x, y) = x^y domain. 'if's divide this domain with respect to
        # monocity.
        le = T.pow(self.lower, exponent)
        ue = T.pow(self.upper, exponent)
        l, u = None, None
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
        """Dot product of Interval(self) vector and a number array (other).

        :param other: Number array to be multiplied.
        :type other: numpy.ndarray.
        """
        # Requires project decision that could be better made after checking
        # number of weights, edges and neurons in considered networks.
        # TODO: Decide how to implement this. Decide whether not to consider
        # batch in this implementation as it will be probably only used for
        # batches.
        raise NotImplementedError

    def max(self, other):
        """Returns interval such that for any numbers (x, y) in a pair of
        corresponding intervals in (self, other) arrays, max(x, y) is in result
        and no other.

        :param other: Interval to be compared.
        :type other: Interval.
        """
        return Interval(T.maximum(self.lower, other.lower),
                        T.maximum(self.upper, other.upper))

    def reshape(self, shape, ndim=None):
        """Reshapes interval tensor like theano Tensor.

        :param shape: Something that can be converted to a symbolic vector of
                      integers.
        :param ndim: The length of the shape. Passing None here means for
                     Theano to try and guess the length of shape.
        """
        return Interval(self.lower.reshape(shape, ndim),
                        self.upper.reshape(shape, ndim))

    def flatten(self, ndim=1):
        """Flattens interval tensor like theano Tensor.

        :param ndim: The number of dimensions in the returned variable.
        :return: Variable with same dtype as x and outdim dimensions.
        :rtype: Variable with the same shape as x in the leading outdim-1
                dimensions, but with all remaining dimensions of x collapsed
                into the last dimension.
        """
        return Interval(self.lower.flatten(ndim),
                        self.upper.flatten(ndim))

    def eval(self, *eval_map):
        """Evaluates interval in terms of theano TensorType eval method.

        :*eval_map: map of Theano variables to be set, just like in
                    theano.tensor.dtensorX.eval method.
        Returns pair (lower, upper) of """
        has_args = (len(eval_map) != 0)
        if has_args:
            eval_map = eval_map[0]
            has_args = (len(eval_map) != 0)
        if not has_args:
            try:
                f = function([], [self.lower, self.upper])
                rlower, rupper = f()
                return (rlower, rupper)
            except:
                return (self.lower, self.upper)
        keys = eval_map.keys()
        values = eval_map.values()
        f = function(keys, [self.lower, self.upper])
        rlower, rupper = f(*values)
        return (rlower, rupper)
