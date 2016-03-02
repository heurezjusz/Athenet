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

if __name__ == '__main__':
    unittest.main(verbosity=2, catchbreak=True)

