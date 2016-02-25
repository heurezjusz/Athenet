"""Auxiliary functions for sparsifying with respect to estimated derivatives.
"""

class Interval(object):
    
    def __init__(lower, upper):
        self.lower = lower
        self.upper = upper

    def __getitem__(self, at):
        return Interval(self.lower[at], self.upper[at])

    def __setitem__(self, at, value):
        self.lower = value.lower[at]
        self.upper = value.upper[at]

    def __len__(self):
        return len(self.lower)

    def __add__(self, other):
        res_lower = self.lower + other.lower
        res_upper = self.upper + other.upper
        return Interval(res_lower, res_upper)

    def __sub__(self, other):
        res_lower = self.lower - other.upper
        res_upper = self.upper - other.lower
        return Interval(res_lower, res_upper)

    def __mul__(self, other):
        ll = self.lower * other.lower
        lu = self.lower * other.upper
        ul = self.upper * other.lower
        uu = self.upper * other.upper
        return Interval(theano.minumum(ll, lu, ul, uu), 
                        theano.maximum(ll, lu, ul, uu))
