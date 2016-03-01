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

    def test_ops1(self):
        # __add__, __sub__, __mul__
        x, y, z, w = T.dmatrices('x', 'y', 'z', 'w')
        i1 = Interval(x, y)
        i2 = Interval(z, w)
        l1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        u1 = l1 * 10 + 1
        l2 = l1 * 10 + 2
        u2 = l1 * 10 + 3
        r_add1 = i1 + i2
        r_add2 = i1 + x
        r_add3 = x + i1
        r_sub1 = i1 - i2
        r_sub2 = x - i1
        r_sub3 = i1 - x
        r_mul1 = i1 * i2
        r_mul2 = i1 * x
        r_mul3 = x * i1
        fres = [r_add1.lower, r_add1.upper, r_add2.lower, r_add2.upper,
                r_add3.lower, r_add3.upper, r_sub1.lower, r_sub1.upper,
                r_sub2.lower, r_sub2.upper, r_sub3.lower, r_sub3.upper,
                r_mul1.lower, r_mul1.upper, r_mul2.lower, r_mul2.upper,
                r_mul3.lower, r_mul3.upper]
        f = function([x, y, z, w], fres)
        add1l, add1u, add2l, add2u, add3l, add3u, \
            sub1l, sub1u, sub2l, sub2u, sub3l, sub3u, \
            mul1l, mul1u, mul2l, mul2u, mul3l, mul3u = f(l1, u1, l2, u2)
        ops_results = [add1l, add1u, add2l, add2u, add3l, add3u,
                       sub1l, sub1u, sub2l, sub2u, sub3l, sub3u,
                       mul1l, mul1u, mul2l, mul2u, mul3l, mul3u]
        r_add1l = l1 + l2
        r_add2l = l1 + l1
        r_add3l = l1 + l1
        r_add1u = u1 + u2
        r_add2u = u1 + l1
        r_add3u = u1 + l1
        r_sub1l = l1 - u2
        r_sub2l = l1 - u1
        r_sub3l = l1 - l1
        r_sub1u = u1 - l2
        r_sub2u = l1 - l1
        r_sub3u = u1 - l1
        r_mul1l = l1 * l2
        r_mul2l = l1 * l1
        r_mul3l = l1 * l1
        r_mul1u = u1 * u2
        r_mul2u = u1 * l1
        r_mul3u = u1 * l1
        results = [r_add1l, r_add1u, r_add2l, r_add2u, r_add3l, r_add3u,
                   r_sub1l, r_sub1u, r_sub2l, r_sub2u, r_sub3l, r_sub3u,
                   r_mul1l, r_mul1u, r_mul2l, r_mul2u, r_mul3l, r_mul3u]
        for i in range(len(ops_results)):
            assert_array_equal(ops_results, results)

    def test_ops2(self):
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

    def test_ops3(self):
        # power
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

        def make_power(exp):
            return i.power(exp)

        powers1 = map(make_power, exponents1)
        powers2 = map(make_power, exponents2)
        powers3 = map(make_power, exponents3)

        def make_lu(power):
            return (power.lower, power.upper)

        lus1 = map(make_lu, powers1)
        lus2 = map(make_lu, powers2)
        lus3 = map(make_lu, powers3)

        def make_function((l, u)):
            return function([x, y], [l, u])

        functions1 = map(make_function, lus1)
        functions2 = map(make_function, lus2)
        functions3 = map(make_function, lus3)

        def make_res1(f):
            return f(*v1)

        def make_res2(f):
            return f(*v2)

        def make_res3(f):
            return f(*v3)

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

    def test_eval(self):
        txl, txu, tyl, tyu = T.dmatrices('xl', 'xu', 'yl', 'yu')
        xl = np.array([[1, 2], [3, 4]])
        xu = np.array([[2, 4], [6, 9]])
        yl = np.array([[-1, -5], [0, 3]])
        yu = np.array([[4, 2], [0, 3]])
        ix = Interval(txl, txu)
        iy = Interval(tyl, tyu)
        iz = ix + iy
        d = {txl: xl, txu: xu, tyl: yl, tyu: yu}
        zl, zu = iz.eval(d)
        assert_array_almost_equal(zl, xl + yl)
        assert_array_almost_equal(zu, xu + yu)
        i2 = Interval(1, 3)
        i2l, i2u = i2.eval()
        assert_equal(i2l, 1)
        assert_equal(i2u, 3)
        i2l, i2u = i2.eval({})
        assert_equal(i2l, 1)
        assert_equal(i2u, 3)

if __name__ == '__main__':
    unittest.main(verbosity=2, catchbreak=True)
