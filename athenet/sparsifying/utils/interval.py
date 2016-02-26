"""Intervals in Theano including special functions for sparsifying."""

from theano import tensor as T
from theano.ifelse import ifelse

class Interval(object):
    
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def __getitem__(self, at):
        """Returns specified slice of interval as a interval.

        at: Coordinates / slice to be taken.
        
        Does not copy data."""
        return Interval(self.lower[at], self.upper[at])

    def __setitem__(self, at, other):
        """Just like Theano set_subtensor function, but as a operator.
        
        at: Coordinates / slice to be set.
        other: Data to be put at 'at'
        
        """
        self.lower = T.set_subtensor(self.lower[at], other.lower)
        self.upper = T.set_subtensor(self.upper[at], other.upper)

    def shape(self):
        """Returns shape of interval. Checks only 'lower' matrix.
        
        Note: does not require self.upper for computations. Therefore it is not
        safe, but faster."""
        return self.lower.shape

    def __add__(self, other):
        """Returns sum of two intervals.
        
        other: Interval to be added."""
        res_lower = self.lower + other.lower
        res_upper = self.upper + other.upper
        return Interval(res_lower, res_upper)

    def __sub__(self, other):
        """Returns difference between two intervals.
        
        other: Interval to be subtracted."""
        res_lower = self.lower - other.upper
        res_upper = self.upper - other.lower
        return Interval(res_lower, res_upper)

    def __mul__(self, other):
        """Returns product of two intervals.
        
        other: Interval to be multiplied."""
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

        #Input check below couses program interrupt if any _has_zero happened.
        #return Interval(switch(self._has_zero(),
        #                       T.constant(1)/T.constant(0),
        #                       T.inv(self.upper)),
        #                T.inv(self.lower))
        return Interval(T.inv(self.upper), T.inv(self.lower))

    def neg(self):
        return Interval(T.neg(self.upper), T.neg(self.lower))

    def exp(self):
        """Returns interval representing the exponential of the interval."""
        return Interval(T.exp(self.lower), T.exp(self.upper))

    def square(self):
        """For interval I, returns I' such that for any x in I, I' contains
        x*x and no other.
        
        e.g [-1, 1].sq() == [0, 1]"""
        lsq = self.lower * self.lower
        usq = self.upper * self.upper
        u = T.maximum(lsq, usq)
        l = T.switch(self._has_zero(), 0, T.minimum(lsq, usq))
        return Interval(l, u)

    def power(self, exponent):
        """For interval i, returns i^exponent.
        
        exponent: Number to be passed as exponent to i^exponent.

        Note: If interval contains some elements lower/equal to 0, exponent
        should be integer."""
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

