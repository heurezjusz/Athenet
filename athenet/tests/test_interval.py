

import unittest
from nose.tools import assert_true, assert_is, assert_equal
from athenet.sparsifying.utils.interval import Interval
import numpy as np
from numpy.testing import assert_array_equal
import theano.tensor as T
from theano import function

class IntervalTest(unittest.TestCase):

    def test_interval_scalar(self):
        x = T.dscalar('x')
        y = T.dscalar('y')
        i = Interval(x, y)
        l = i.lower
        u = i.upper
        assert_is(x, l)
        assert_is(y, u)
        z = u - l
        f = function([x, y], z)
        assert_equal(f(1.1, 3.2), 2.1)

    def test_interval_matrix(self):
        x = T.dmatrix('x')
        y = T.dmatrix('y')
        i = Interval(x, y)
        l = i.lower
        u = i.upper
        assert_is(x, l)
        assert_is(y, u)
        z = u + l
        f = function([x, y], z)
        res_z = f([[1, 2], [3, 4]], [[10, 20], [30, 40]])
        res = np.array([[11., 22.], [33., 44.]])
        assert_array_equal(res_z, res)

    def test_getitem(self):
        x = T.dmatrix('x')
        y = T.dmatrix('y')
        i = Interval(x, y)
        i0 = i[0, 0]
        i1 = i[1, 1]
        i2 = i[2, 2]
        l0, l1, l2 = i0.lower, i1.lower, i2.lower
        u0, u1, u2 = i0.upper, i1.upper, i2.upper
        f = function([x, y], [l0, l1, l2, u0, u1, u2])
        ex_x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        ex_y = ex_x * 10
        [rl0, rl1, rl2, ru0, ru1, ru2] = f(ex_x, ex_y)
        assert_equal(rl0, 1)
        assert_equal(rl1, 5)
        assert_equal(rl2, 9)
        assert_equal(ru0, 10)
        assert_equal(ru1, 50)
        assert_equal(ru2, 90)

if __name__ == '__main__':
    unittest.main(verbosity=2, catchbreak=True)
