"""Template class with arithmetic operations that can be passed through neural
network.

All classes that are being used for derest should inherit from this class."""


class Numlike(object):
    """Template class with arithmetic operations that can be passed through
    neural network.

    All classes that are being used for derest should inherit from this
    class."""

    def __init__(self):
        """Create numlike."""
        pass

    def __getitem__(self, at):
        """Returns specified slice of numlike.

        :at: Coordinates / slice to be taken.
        """
        raise NotImplementedError

    def __setitem__(self, at, other):
        """Just like Theano set_subtensor function, but as a operator.

        :at: Coordinates / slice to be set.
        :other: Data to be put at 'at'.
        """
        raise NotImplementedError

    @property
    def shape(self):
        """Returns shape of numlike."""
        raise NotImplementedError

    def __add__(self, other):
        """Returns sum of two numlikes.

        :other: numlike.
        """
        raise NotImplementedError

    def __sub__(self, other):
        """Returns difference between two numlikes.

        :other: numlike to be subtracted.
        """
        raise NotImplementedError

    def __mul__(self, other):
        """Returns product of two numlikes.

        :other: numlike to be multiplied.
        """
        raise NotImplementedError

    def __div__(self, other):
        """Returns quotient of self and other."""
        raise NotImplementedError

    def __rdiv__(self, other):
        """Returns quotient of other and self.

        :param other: dividend
        :type other: float
        :rtype: Nplike
        .. warning:: divisor (self) should not contain zero, other must be
                     float
        """
        raise NotImplementedError

    def reciprocal(self):
        """Returns reciprocal of the numlike."""
        raise NotImplementedError

    def neg(self):
        """Returns (-1) * numlike."""
        raise NotImplementedError

    def exp(self):
        """Returns numlike representing the exponential of the numlike."""
        raise NotImplementedError

    def square(self):
        """Returns square of the numlike."""
        raise NotImplementedError

    def power(self, exponent):
        """For numlike N, returns N^exponent.

        :param float exponent: Number to be passed as exponent to N^exponent.
        """
        raise NotImplementedError

    def dot(self, other):
        """Dot product of numlike vector and a other.

        :param unspecified other: second dot param, type to be specified
        """
        raise NotImplementedError

    def max(self, other):
        """Returns maximum of self and other.

        :param unspecified other: second masx param, type to be specified
        """
        raise NotImplementedError

    def amax(self, axis=None, keepdims=False):
        """Returns maximum of a Numlike along an axis.

        Works like theano.tensor.max

        :param axis: axis along which max is evaluated
        :param Boolean keepdims: whether flattened dimensions should remain
        """
        raise NotImplementedError

    def reshape(self, shape):
        """Reshapes numlike tensor like theano Tensor.

        :param integer tuple shape: shape to be set
        """
        raise NotImplementedError

    def flatten(self):
        """Flattens numlike tensor like theano Tensor."""
        raise NotImplementedError

    def sum(self, axis=None, dtype=None, keepdims=False):
        """Vector operation like in numpy.ndarray.

        :param axis: axis along which this function sums
        :param numeric type or None dtype: just like dtype argument in
                                   theano.tensor.sum
        :param Boolean keepdims: Whether to keep squashed dimensions of size 1
        :type axis: integer, tuple of integers or None

        """
        raise NotImplementedError

    def abs(self):
        """Returns absolute value of Numlike."""
        raise NotImplementedError

    @property
    def T(self):
        """Vector operation like in numpy.ndarray"""
        raise NotImplementedError

    @staticmethod
    def from_shape(shp, neutral=True):
        """Returns Numlike of given shape.

        :param integer tuple shp: shape to be set
        :param Boolean neutral: whether created Numlike should have neutral
                        values or significant values.
        """
        raise NotImplementedError

    def eval(self):
        """Returns some readable form of stored value."""
        raise NotImplementedError
