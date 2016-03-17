"""numpy.ndarray packed in Numlike interface."""

from athenet.sparsifying.utils.numlike import Numlike
import numpy


class Nplike(Numlike):

    def __init__(self, value):
        self.value = value

    def __getitem__(self, at):
        return Nplike(self.value[at])

    def __setitem__(self, at, other):
        self.value[at] = other.value[at]

    def __repr__(self):
        """Standard repr method."""
        return repr(self.value)

    def __str__(self):
        """"Standard str method."""
        return str(self.value)

    @property
    def shape(self):
        return self.value.shape

    def __add__(self, other):
        """Note: __radd__ is not being invoked if :other: is numpy array."""
        if isinstance(other, Nplike):
            return Nplike(self.value + other.value)
        else:
            return Nplike(self.value + other)

    __radd__ = __add__

    def __sub__(self, other):
        return Nplike(self.value - other.value)

    def __rsub__(self, other):
        return Nplike(other.value - self.value)

    def __mul__(self, other):
        """Note: __radd__ is not being invoked if :other: is numpy array."""
        if isinstance(other, Nplike):
            return Nplike(self.value * other.value)
        else:
            return Nplike(self.value * other)

    __rmul__ = __mul__

    def __div__(self, other):
        return Nplike(self.value / other.value)

    def __rdiv__(self, other):
        return Nplike(other.value / self.value)

    def reciprocal(self):
        return Nplike(numpy.reciprocal(self.value))

    def neg(self):
        return Nplike(-self.value)

    def exp(self):
        return Nplike(numpy.exp(self.value))

    def square(self):
        return Nplike(numpy.square(self.value))

    def power(self, exponent):
        return Nplike(numpy.power(self.value, exponent))

    def dot(self, other):
        """other must be numpy array."""
        return Nplike(numpy.dot(self.value, other))

    def max(self, other):
        if isinstance(other, Nplike):
            return Nplike(numpy.maximum(self.value, other.value))
        else:
            return Nplike(numpy.maximum(self.value, other))

    def amax(self, axis=None, keepdims=False):
        return Nplike(numpy.amax(self.value, axis=axis, keepdims=keepdims))

    def reshape(self, shape):
        return Nplike(self.value.reshape(shape))

    def flatten(self):
        return Nplike(self.value.flatten())

    def sum(self, axis=None, dtype=None, keepdims=False):
        """Vector operation like in numpy.ndarray."""
        s = self.value.sum(axis=axis, dtype=dtype, keepdims=keepdims)
        if isinstance(s, numpy.ndarray):
            return Nplike(s)
        else:
            return Nplike(numpy.array([s]))

    @property
    def T(self):
        """Vector operation like in numpy.ndarray."""
        return Nplike(self.value.T)

    @staticmethod
    def from_shape(shp, neutral=True):
        if neutral:
            return Nplike(numpy.zeros(shp))
        else:
            return Nplike(numpy.ones(shp))

    def eval(self):
        return self.value
