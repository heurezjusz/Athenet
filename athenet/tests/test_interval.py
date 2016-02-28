"""Testing athenet.sparsifying.utils.interval.Interval class with its methods.
"""

import unittest
from nose.tools import assert_true, assert_is, assert_equal
from athenet.sparsifying.utils.interval import Interval
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
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
        radd2 = i1 + x
        rsub = i1 - i2
        rmul = i1 * i2
        fres = [radd.lower, radd.upper, radd2.lower, radd2.upper, rsub.lower,
                rsub.upper, rmul.lower, rmul.upper]
        f = function([x, y, z, w], fres)
        addl, addu, add2l, add2u, subl, subu, mull, mulu = f(l1, u1, l2, u2)
        raddl = l1 + l2
        raddu = u1 + u2
        radd2l = l1 + l1
        radd2u = u1 + l1
        rsubl = l1 - u2
        rsubu = u1 - l2
        rmull = l1 * l2
        rmulu = u1 * u2
        assert_array_equal(addl, raddl)
        assert_array_equal(addu, raddu)
        assert_array_equal(add2l, radd2l)
        assert_array_equal(add2u, radd2u)
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
        v1l = np.array([-3, -2, -1, -2, 0.5, 0.5, 1, 2])
        v1u = np.array([-2, -1, -0.5, -0.5, 2, 1, 2, 3])
        v2l = np.array([1, 2])
        v2u = np.array([3, 4])
        v3l = np.array([-2., -2., -2., -1., -1., -1., -0.5, -0.5, -0.5])
        v3u = np.array([0.5, 1., 2., 0.5, 1., 2., 0.5, 1., 2.])
        v1 = (v1l, v1u)
        v2 = (v2l, v2u)
        v3 = (v3l, v3u)
        exponents1 = [-3, -2, 2, 3]
        exponents2 = [-2.5, -2., 2., 2.5]
        exponents3 = [2, 3]
        make_power = lambda exp: i.power(exp)
        powers1 = map(make_power, exponents1)
        powers2 = map(make_power, exponents2)
        powers3 = map(make_power, exponents3)
        make_lu = lambda power: (power.lower, power.upper)
        lus1 = map(make_lu, powers1)
        lus2 = map(make_lu, powers2)
        lus3 = map(make_lu, powers3)
        make_function = lambda (l, u): function([x, y], [l, u])
        functions1 = map(make_function, lus1)
        functions2 = map(make_function, lus2)
        functions3 = map(make_function, lus3)
        make_res1 = lambda f: f(*v1)
        make_res2 = lambda f: f(*v2)
        make_res3 = lambda f: f(*v3)
        res1 = map(make_res1, functions1)
        res2 = map(make_res2, functions2)
        res3 = map(make_res3, functions3)
        ans1l = [np.array([4., 1., 0.25, 0.25, 0.25, 0.25, 1., 4.]),
                 np.array([-27., -8., -1., -8., 0.125, 0.125, 1., 8.])]
        ans1u = [np.array([9., 4., 1., 4., 4., 1., 4., 9.]),
                 np.array([-8., -1., -0.125, -0.125, 8., 1., 8., 27.])]
        ans1l = [np.reciprocal(ans1u[1]), np.reciprocal(ans1u[0])] + ans1l
        ans1u = [np.reciprocal(ans1l[3]), np.reciprocal(ans1l[2])] + ans1u
        ans2l = [np.array([1., 4.]), np.array([1., 2. ** 2.5])]
        ans2u = [np.array([9., 16.]), np.array([3. ** 2.5, 4. ** 2.5])]
        ans2l = [np.reciprocal(ans2u[1])] + [np.reciprocal(ans2u[0])] + ans2l
        ans2u = [np.reciprocal(ans2l[3])] + [np.reciprocal(ans2l[2])] + ans2u
        ans3l = [np.array([0.] * 9),
                 np.array([-8., -8., -8., -1., -1., -1., -0.125, -0.125,
                          -0.125])]
        ans3u = [np.array([4., 4., 4., 1., 1., 4., 0.25, 1., 4.]),
                 np.array([0.125, 1., 8., 0.125, 1., 8., 0.125, 1., 8.])]
        for i in range(4):
            assert_array_almost_equal(res1[i][0], ans1l[i])
            assert_array_almost_equal(res1[i][1], ans1u[i])
            assert_array_almost_equal(res2[i][0], ans2l[i])
            assert_array_almost_equal(res2[i][1], ans2u[i])
        for i in range(2):
            assert_array_almost_equal(res3[i][0], ans3l[i])
            assert_array_almost_equal(res3[i][1], ans3u[i])

if __name__ == '__main__':
    unittest.main(verbosity=2, catchbreak=True)
