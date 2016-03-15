"""Testing athenet.sparsifying.utils.interval.Interval class with its methods.
"""

import numpy as np
import unittest
from nose.tools import raises
from athenet.sparsifying.utils.numlike import Numlike


class NumlikeTest(unittest.TestCase):

    def test_init(self):
        n = Numlike()

    @raises(NotImplementedError)
    def test_getitem(self):
        a = Numlike()[0]

    @raises(NotImplementedError)
    def test_setitem(self):
        Numlike()[0] = 0.0

    @raises(NotImplementedError)
    def test_shape(self):
        shp = Numlike().shape()

    @raises(NotImplementedError)
    def test_add(self):
        res = Numlike() + Numlike()

    @raises(NotImplementedError)
    def test_radd(self):
        res = 1.0 + Numlike()

    @raises(NotImplementedError)
    def test_sub(self):
        res = Numlike() - 1.0

    @raises(NotImplementedError)
    def test_rsub(self):
        res = 1.0 - Numlike()

    @raises(NotImplementedError)
    def test_mul(self):
        res = Numlike() * 3.0

    @raises(NotImplementedError)
    def test_rmul(self):
        res = 3.0 * Numlike()

    @raises(NotImplementedError)
    def test_div(self):
        res = Numlike() / 5.0

    @raises(NotImplementedError)
    def test_reciprocal(self):
        res = Numlike().reciprocal()

    @raises(NotImplementedError)
    def test_neg(self):
        res = Numlike().neg()

    @raises(NotImplementedError)
    def test_exp(self):
        res = Numlike().exp()

    @raises(NotImplementedError)
    def test_square(self):
        res = Numlike().square()

    @raises(NotImplementedError)
    def test_power(self):
        res = Numlike().power(3.0)

    @raises(NotImplementedError)
    def test_dot(self):
        w = np.array([[1, 2], [3, 4]])
        res = Numlike().dot(w)

    @raises(NotImplementedError)
    def test_max(self):
        res = Numlike().max(Numlike())

    @raises(NotImplementedError)
    def test_reshape(self):
        res = Numlike().reshape((1, 2, 3))

    @raises(NotImplementedError)
    def test_flatten(self):
        res = Numlike().flatten()

    @raises(NotImplementedError)
    def test_sum(self):
        res = Numlike().sum(0)

    @raises(NotImplementedError)
    def test_T(self):
        res = Numlike().T

    @raises(NotImplementedError)
    def test_from_shape1(self):
        res = Numlike.from_shape((3, 4))

    @raises(NotImplementedError)
    def test_from_shape2(self):
        res = Numlike.from_shape((3, 4), neutral=True)

    @raises(NotImplementedError)
    def test_from_shape3(self):
        res = Numlike.from_shape((3, 4), neutral=False)

if __name__ == '__main__':
    unittest.main(verbosity=2, catchbreak=True)

