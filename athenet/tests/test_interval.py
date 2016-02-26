"""Testing athenet.sparsifying.utils.interval.Interval class with its methods.
"""

import unittest
from nose.tools import assert_true, assert_is, assert_equal
from athenet.sparsifying.utils.interval import Interval
import numpy as np
from numpy.testing import assert_array_equal
import theano.tensor as T
from theano import function

class IntervalTest(unittest.TestCase):

    def test_interval_scalar(self):
        x, y = T.dscalars('x', 'y')
        i = Interval(x, y)
        l, u = i.lower, i.upper
        assert_is(x, l)
        assert_is(y, u)
        z = u - l
        f = function([x, y], z)
        assert_equal(f(1.1, 3.2), 2.1)

    def test_interval_matrix(self):
        x, y = T.dmatrices('x', 'y')
        i = Interval(x, y)
        l, u = i.lower, i.upper
        assert_is(x, l)
        assert_is(y, u)
        z = u + l
        f = function([x, y], z)
        res_z = f([[1, 2], [3, 4]], [[10, 20], [30, 40]])
        res = np.array([[11., 22.], [33., 44.]])
        assert_array_equal(res_z, res)

    def test_getitem(self):
        x, y = T.dmatrices('x', 'y')
        i = Interval(x, y)
        i0, i1, i2 = i[0, 0], i[1, 1], i[2, 2]
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

    def test_setitem(self):
        x, y, z, w = T.dmatrices('x', 'y', 'z', 'w')
        i1 = Interval(x, y)
        i2 = Interval(z, w)
        i1[:, 1:3] = i2
        f = function([x, y, z, w], [i1.lower, i1.upper])
        ex_x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        ex_z = np.array([[20, 30], [50, 60], [80, 90]])
        ex_y = 100 * ex_x
        ex_w = 100 * ex_z
        l, u = f(ex_x, ex_y, ex_z, ex_w)
        rl = np.array([[1., 20., 30.], [4., 50., 60.], [7., 80., 90.]])
        ru = rl * 100
        assert_array_equal(l, rl)
        assert_array_equal(u, ru)

    def test_shape(self):
        x, y = T.dmatrices('x', 'y')
        i1 = Interval(x, y)
        ex_x = np.array([[2, 3], [5, 6], [8, 9]])
        ex_y = 10 * ex_x
        shp = i1.shape()
        rshp = shp.eval({x: ex_x})
        assert_equal(len(rshp), 2)
        assert_equal(rshp[0], 3)
        assert_equal(rshp[1], 2)

    def test_ops(self):
        # __add__, __sub__, __mul__
        x, y, z, w = T.dmatrices('x', 'y', 'z', 'w')
        i1 = Interval(x, y)
        i2 = Interval(z, w)
        l1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        u1 = l1 * 10
        l2 = l1 * 100
        u2 = l1 * 1000
        radd = i1 + i2
        rsub = i1 - i2
        rmul = i1 * i2
        fres = [radd.lower, radd.upper, rsub.lower, rsub.upper, rmul.lower,
                rmul.upper]
        f = function([x, y, z, w], fres)
        addl, addu, subl, subu, mull, mulu = f(l1, u1, l2, u2)
        raddl = l1 + l2
        raddu = u1 + u2
        rsubl = l1 - u2
        rsubu = u1 - l2
        rmull = l1 * l2
        rmulu = u1 * u2
        assert_array_equal(addl, raddl)
        assert_array_equal(addu, raddu)
        assert_array_equal(subl, rsubl)
        assert_array_equal(subu, rsubu)
        assert_array_equal(mull, rmull)
        assert_array_equal(mulu, rmulu)

    def test_op2(self):
        # reciprocal, neg, exp, sq
        x, y = T.dmatrices('x', 'y')
        i = Interval(x, y)
        reciprocal = i.reciprocal()
        neg = i.neg()
        exp = i.exp()
        sq = i.square()
        recl, recu = reciprocal.lower, reciprocal.upper
        negl, negu = neg.lower, neg.upper
        expl, expu = exp.lower, exp.upper
        sql, squ = sq.lower, sq.upper
        l = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        u = l + 10.0
        f = function([x, y], [recl, recu, negl, negu, expl, expu, sql, squ])
        rrecl, rrecu, rnegl, rnegu, rexpl, rexpu, rsql, rsqu = f(l, u)
        threcl = np.reciprocal(u)
        threcu = np.reciprocal(l)
        thnegl = -u
        thnegu = -l
        thexpl = np.exp(l)
        thexpu = np.exp(u)
        thsql = np.square(l)
        thsqu = np.square(u)
        assert_array_equal(rrecl, threcl)
        assert_array_equal(rrecu, threcu)
        assert_array_equal(rnegl, thnegl)
        assert_array_equal(rnegu, thnegu)
        assert_array_equal(rexpl, thexpl)
        assert_array_equal(rexpu, thexpu)
        assert_array_equal(rsql, thsql)
        assert_array_equal(rsqu, thsqu)

    def test_op3(self):
        #power
        x, y = T.vectors('x', 'y')
        i = Interval(x, y)
        v1l = np.array([-3, -2, -1, 0, 1, 2, 3])
        v1u = v1l + 2
        v2l = np.array([0, 1, 2])
        v2u = v2l + 2
        v3l = np.array([1, 2])
        v3u = v3l + 2
        fr1 = i.power(-2.5)
        fr2 = i.power(-2.)
        fr3 = i.power(2.)
        fr4 = i.power(2.5)
        fr5 = i.power(-2)
        fr6 = i.power(-3)
        fr7 = i.power(2)
        fr8 = i.power(3)
        f1 = function([x, y], r1)
        f2 = function([x, y], r2)
        f3 = function([x, y], r3)
        f4 = function([x, y], r4)
        f5 = function([x, y], r5)
        f6 = function([x, y], r6)
        f7 = function([x, y], r7)
        f8 = function([x, y], r8)
        r1 = f1(v3l, v3u)
        r2 = f2(v3l, v3u)
        r3 = f3(v3l, v3u)
        r4 = f4(v3l, v3u)






if __name__ == '__main__':
    unittest.main(verbosity=2, catchbreak=True)
