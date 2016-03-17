"""Testing athenet.sparsifying.utils.interval.Interval class with its methods.
"""

import unittest
from nose.tools import raises, assert_true
from athenet.sparsifying.utils import Nplike
import numpy as np
from numpy.testing import assert_array_equal as are, \
    assert_array_almost_equal as arae

A = np.array


class NumlikeTest(unittest.TestCase):

    def prepare(self, shp):
        a = np.arange(np.prod(shp)).reshape(shp)
        n = Nplike(a)
        return (a, n)

    def test_init(self):
        (a, n) = self.prepare((2, 3))

    def test_getitem(self):
        (a, n) = self.prepare((2, 3))
        n2 = n[1:2, 1:3]
        are(n2.eval(), A([[4, 5]]))

    def test_setitem(self):
        (a, n) = self.prepare((2, 3))
        n[0:2, 0:2] = Nplike(A([[11, 12], [13, 14]]))
        are(n.eval(), A([[11, 12, 2], [13, 14, 5]]))

    def test_shape(self):
        (a, n) = self.prepare((2, 3))
        shp = n.shape
        assert_true(shp == (2, 3))

    def test_add(self):
        (a, n) = self.prepare((2, 3))
        n2 = Nplike(A([[3, 2, 2], [4, 5, 4]]))
        are((n + n2).eval(), A([[3, 3, 4], [7, 9, 9]]))

    def test_sub(self):
        (a, n) = self.prepare((2, 3))
        n2 = Nplike(A([[3, 2, 2], [4, 5, 4]]))
        are((n - n2).eval(), A([[-3, -1, 0], [-1, -1, 1]]))

    def test_mul(self):
        (a, n) = self.prepare((2, 3))
        n2 = Nplike(A([[3, 2, 2], [4, 5, 4]]))
        are((n * n2).eval(), A([[0, 2, 4], [12, 20, 20]]))

    def test_div(self):
        (a, n) = self.prepare((2, 3))
        n2 = Nplike(A([[3, 2, 2], [4, 5, 4]]))
        are((n + n2).eval(), A([[3, 3, 4], [7, 9, 9]]))

    def test_reciprocal(self):
        a = (np.arange(6) + 1.0).reshape((2, 3))
        n = Nplike(a)
        arae(n.reciprocal().eval(), 1.0 / a)

    def test_neg(self):
        (a, n) = self.prepare((2, 3))
        arae(-a, n.neg().eval())

    def test_exp(self):
        a = (np.arange(6) + 0.0).reshape((2, 3))
        n = Nplike(a)
        arae(np.exp(a), n.exp().eval())

    def test_square(self):
        (a, n) = self.prepare((2, 3))
        arae(a * a, n.square().eval())

    def test_power(self):
        (a, n) = self.prepare((2, 3))
        e = 3.1415
        arae(np.power(a, e), n.power(e).eval())

    def test_dot(self):
        (a, n) = self.prepare((2, 3))
        (b, m) = self.prepare((3, 4))
        p = n.dot(b).eval()
        arae(p, A([[20, 23, 26, 29], [56, 68, 80, 92]]))

    def test_max(self):
        (a, n) = self.prepare((2, 3))
        b = A([[3, 0, 3], [0, 3, -123]])
        m = Nplike(b)
        arae(n.max(b).eval(), A([[3, 1, 3], [3, 4, 5]]))

    def test_amax(self):
        (a, n) = self.prepare((2, 3))
        m = n.amax(axis=0, keepdims=True)
        m2 = n.amax(axis=1, keepdims=True)
        b1 = m.eval()
        b2 = m2.eval()
        arae(b1, A([[3, 4, 5]]))
        arae(b2, A([[2], [5]]))

    def test_reshape(self):
        (a, n) = self.prepare((2, 3))
        m = n.reshape((6, 1))
        arae(m.eval(), A([[0], [1], [2], [3], [4], [5]]))

    def test_flatten(self):
        (a, n) = self.prepare((2, 3))
        m = n.flatten()
        arae(m.eval(), A([0, 1, 2, 3, 4, 5]))

    def test_sum(self):
        (a, n) = self.prepare((2, 3))
        s = n.sum()
        arae(s.eval(), A([15]))

    def test_T(self):
        (a, n) = self.prepare((2, 3))
        m = n.T
        arae(m.eval(), A([[0, 3], [1, 4], [2, 5]]))

    def test_from_shape1(self):
        n = Nplike.from_shape((2, 3), neutral=True)
        arae(n.eval(), np.zeros((2, 3)))

    def test_from_shape2(self):
        n = Nplike.from_shape((2, 3), neutral=False)
        arae(n.eval(), np.ones((2, 3)))

if __name__ == '__main__':
    unittest.main(verbosity=2, catchbreak=True)
