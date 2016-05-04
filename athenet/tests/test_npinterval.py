from athenet.algorithm.numlike import NpInterval
from unittest import TestCase, main
from random import randrange, uniform
from itertools import product
import numpy as np


def _random_shape(n = None, limit = None):
    if n is None:
        n = randrange(1, 7)
    result = None
    if limit is None:
        limit = 10 ** 4
    size = 1
    for i in xrange(n):
        l = randrange(1, 10)
        if result is None:
            result = (l,)
        else:
            result += (l,)
        size *= l
        if size >= limit:
            break
    while len(result) < n:
        result += (1,)
    return result


def _random_npinterval(shape = None, dim = None):
    if shape is None:
        shape = _random_shape(dim)
    r1 = np.random.rand(*shape) * 20 - 10
    r2 = np.random.rand(*shape) * 20 - 10
    return NpInterval(np.minimum(r1, r2), np.maximum(r1, r2))

def _rand_from_npinterval(I):
    l = I.lower
    u = I.upper
    return (l + np.random.rand(*I.shape) * (u - l))


class TestLocalFunctions(TestCase):
    def test_rand_from(self):
        for _ in xrange(100):
            I = _random_npinterval()
            r = _rand_from_npinterval(I)
            self.assertTrue((I.lower <= r).all())
            self.assertTrue((r <= I.upper).all())


class TestShape(TestCase):
    def _run_test(self, shape):
        i = NpInterval(np.zeros(shape), np.ones(shape))
        self.assertEquals(shape, i.shape)

    def test_shape(self):
        for i in xrange(100):
            self._run_test(_random_shape())


class TestMultiplying(TestCase):
    def test_case(self):
        al = np.asarray([[1, -2, -1], [-42, -5, -1]])
        au = np.asarray([[2, -1,  1], [  4, -4, 7]])
        A = NpInterval(al, au)

        bl = np.asarray([[1, 1, 4], [-1, -2, -13]])
        bu = np.asarray([[2, 2, 5], [-1, -1,   1]])
        B = NpInterval(bl, bu)

        rl = np.asarray([[1, -4, -5], [-4,  4, -91]])
        ru = np.asarray([[4, -1,  5], [42, 10,  13]])

        R = A * B
        self.assertTrue((rl == R.lower).all())
        self.assertTrue((ru == R.upper).all())

        R = B
        R *= A
        self.assertTrue((rl == R.lower).all())
        self.assertTrue((ru == R.upper).all())

    def _check_result(self, A, B, R):
        al, au = A.lower[0], A.upper[0]
        bl, bu = B.lower[0], B.upper[0]
        rl, ru = R.lower[0], R.upper[0]
        for a, b in product(xrange(al, au), xrange(bl, bu)):
            self.assertTrue(rl <= a * b <= ru)

    def test_correct(self):
        for i in xrange(100):
            l = [randrange(-10, 10) for j in xrange(4)]
            A = NpInterval(np.asarray([l[0]]), np.asarray([l[1]]))
            B = NpInterval(np.asarray([l[2]]), np.asarray([l[3]]))

            if A.lower[0] > A.upper[0]:
                A.lower, A.upper = A.upper, A.lower
            if B.lower[0] > B.upper[0]:
                B.lower, B.upper = B.upper, B.lower

            self._check_result(A, B, B * A)

    def test_shape(self):
        for i in xrange(100):
            shape = _random_shape()
            A = NpInterval(np.ones(shape), 2 * np.ones(shape))
            B = NpInterval(np.ones(shape) * 2, np.ones(shape) * 3)
            R = A * B
            self.assertEqual(R.shape, shape)


class TestAdding(TestCase):
    def test_case(self):
        al = np.asarray([[1, -2, -1], [-4, -5, -1]])
        au = np.asarray([[2, -1,  1], [42, -4, 7]])
        A = NpInterval(al, au)

        bl = np.asarray([[1, 1, 4], [-1, -2, -13]])
        bu = np.asarray([[2, 2, 5], [-1, -1,   1]])
        B = NpInterval(bl, bu)

        rl = np.asarray([[2, -1, 3], [-5, -7, -14]])
        ru = np.asarray([[4,  1, 6], [41, -5,  8]])

        R = A + B
        self.assertTrue((rl == R.lower).all())
        self.assertTrue((ru == R.upper).all())

        R = B
        R += A
        self.assertTrue((rl == R.lower).all())
        self.assertTrue((ru == R.upper).all())

    def _check_result(self, A, B, R):
        al, au = A.lower[0], A.upper[0]
        bl, bu = B.lower[0], B.upper[0]
        rl, ru = R.lower[0], R.upper[0]
        for a, b in product(xrange(al, au), xrange(bl, bu)):
            self.assertTrue(rl <= a + b <= ru)
        self.assertTrue(al + bl - 1 < rl)
        self.assertTrue(au + bu + 1 > ru)

    def test_correct(self):
        for i in xrange(100):
            l = [randrange(-10, 10) for j in xrange(4)]
            A = NpInterval(np.asarray([l[0]]), np.asarray([l[1]]))
            B = NpInterval(np.asarray([l[2]]), np.asarray([l[3]]))

            if A.lower[0] > A.upper[0]:
                A.lower, A.upper = A.upper, A.lower
            if B.lower[0] > B.upper[0]:
                B.lower, B.upper = B.upper, B.lower

            self._check_result(A, B, B + A)

    def test_shape(self):
        for i in xrange(100):
            shape = _random_shape()
            A = NpInterval(np.ones(shape), 2 * np.ones(shape))
            B = NpInterval(np.ones(shape) * 2, np.ones(shape) * 3)
            R = A + B
            self.assertEqual(R.shape, shape)


class TestSub(TestCase):
    def test_case(self):
        al = np.asarray([[1, -2, -1], [-4, -5, -1]])
        au = np.asarray([[2, -1,  1], [42, -4, 7]])
        A = NpInterval(al, au)

        bl = np.asarray([[1, 1, 4], [-1, -2, -13]])
        bu = np.asarray([[2, 2, 5], [-1, -1,   1]])
        B = NpInterval(bl, bu)

        rl = np.asarray([[-1, -4, -6], [-3, -4, -2]])
        ru = np.asarray([[ 1, -2, -3], [43, -2, 20]])

        R = A - B
        self.assertTrue((rl == R.lower).all())
        self.assertTrue((ru == R.upper).all())

    def _check_result(self, A, B, R):
        al, au = A.lower[0], A.upper[0]
        bl, bu = B.lower[0], B.upper[0]
        rl, ru = R.lower[0], R.upper[0]
        for a, b in product(xrange(al, au), xrange(bl, bu)):
            self.assertTrue(rl <= a - b <= ru)

    def test_correct(self):
        for i in xrange(100):
            l = [randrange(-10, 10) for j in xrange(4)]
            A = NpInterval(np.asarray([l[0]]), np.asarray([l[1]]))
            B = NpInterval(np.asarray([l[2]]), np.asarray([l[3]]))

            if A.lower[0] > A.upper[0]:
                A.lower, A.upper = A.upper, A.lower
            if B.lower[0] > B.upper[0]:
                B.lower, B.upper = B.upper, B.lower

            self._check_result(A, B, A - B)

    def test_shape(self):
        for i in xrange(100):
            shape = _random_shape()
            A = NpInterval(np.ones(shape), 2 * np.ones(shape))
            B = NpInterval(np.ones(shape) * 2, np.ones(shape) * 3)
            R = A - B
            self.assertEqual(R.shape, shape)



class TestSquare(TestCase):
    def test_case(self):
        al = np.asarray([[1, -2, -1], [-42, -5, -1]])
        au = np.asarray([[2, -1,  1], [  4, -4, 7]])
        A = NpInterval(al, au)

        rl = np.asarray([[1, 1, 0], [   0, 16, 0]])
        ru = np.asarray([[4, 4, 1], [1764, 25, 49]])

        R = A.square()
        self.assertTrue((rl == R.lower).all())
        self.assertTrue((ru == R.upper).all())

    def _check_result(self, A, R):
        al, au = A.lower[0], A.upper[0]
        rl, ru = R.lower[0], R.upper[0]
        for a in xrange(al, au):
            self.assertTrue(rl <= a * a <= ru)
        bigger = max(al*al, au*au) + 1
        self.assertTrue(bigger > ru)
        self.assertTrue(rl > -1)

    def test_correct(self):
        for i in xrange(100):
            a, b = randrange(-10, 10), randrange(-10, 10)
            if a > b:
                a, b = b, a
            A = NpInterval(np.asarray([a]), np.asarray([b]))
            self._check_result(A, A.square())

    def test_shape(self):
        for i in xrange(100):
            shape = _random_shape()
            A = NpInterval(np.ones(shape), 2 * np.ones(shape))
            self.assertEqual(A.square().shape, shape)


class TestGetSetitem(TestCase):
    def test_1D(self):
        n = 100
        I = NpInterval(np.zeros((n,)), np.zeros((n,)))
        for i in xrange(n):
            I[i] = NpInterval(np.asarray([-i]),np.asarray([i]))
        for i in xrange(n):
            self.assertEquals(I[i].lower, -i)
            self.assertEquals(I[i].upper, i)

    def test_2D(self):
        n = 100
        I = NpInterval(np.zeros((n,n)), np.zeros((n,n)))
        for i, j in product(xrange(n), xrange(n)):
            I[i][j] = NpInterval(np.asarray([i ^ (j**2) - 42]),
                                 np.asarray([i**2 + j**3 / 7]))
        for i, j in product(xrange(n), xrange(n)):
            self.assertEquals(I[i][j].lower, i ^ (j**2) - 42)
            self.assertEquals(I[i][j].upper, i**2 + j**3 / 7)

    def test_3D(self):
        n = 10
        I = NpInterval(np.zeros((n, n, n)), np.zeros((n, n, n)))
        for i, j, k in product(xrange(n), xrange(n), xrange(n)):
            I[i][j][k] = NpInterval(np.asarray([i + j - k ^ 67]),
                                    np.asarray([i * j + 42 * k]))
        for i, j, k in product(xrange(n), xrange(n), xrange(n)):
            self.assertEquals(I[i][j][k].lower, i + j - k ^ 67)
            self.assertEquals(I[i][j][k].upper, i * j + 42 * k)

    def test_4D(self):
        n = 10
        I = NpInterval(np.zeros((n, n, n, n)), np.zeros((n, n, n, n)))
        for i, j, k, l in product(xrange(n), xrange(n), xrange(n), xrange(n)):
            I[i][j][k][l] = NpInterval(np.asarray([i*l ^ j*k]),
                                        np.asarray([(i*j ^ l*k) + 1000]))
        for i, j, k, l in product(xrange(n), xrange(n), xrange(n), xrange(n)):
            self.assertEquals(I[i][j][k][l].lower, i*l ^ j*k)
            self.assertEquals(I[i][j][k][l].upper, (i*j ^ l*k) + 1000)


class TestAntiadd(TestCase):
    def test_case(self):
        al = np.asarray([[1, -2, -1], [-4, -5, -1]])
        au = np.asarray([[2, -1,  1], [42, -4, 7]])
        A = NpInterval(al, au)

        bl = np.asarray([[1, 1, 4], [-1, -2, -13]])
        bu = np.asarray([[2, 2, 5], [-1, -1,   1]])
        B = NpInterval(bl, bu)

        R = (A + B).antiadd(B)
        self.assertTrue((A.lower == R.lower).all())
        self.assertTrue((A.upper == R.upper).all())
        R = (A + B).antiadd(A)
        self.assertTrue((B.lower == R.lower).all())
        self.assertTrue((B.upper == R.upper).all())

    def test_correct(self):
        for _ in xrange(100):
            l = [randrange(-10, 10) for j in xrange(4)]
            A = NpInterval(np.asarray([l[0]]), np.asarray([l[1]]))
            B = NpInterval(np.asarray([l[2]]), np.asarray([l[3]]))

            if A.lower[0] > A.upper[0]:
                A.lower, A.upper = A.upper, A.lower
            if B.lower[0] > B.upper[0]:
                B.lower, B.upper = B.upper, B.lower

            R = (A + B).antiadd(B)
            self.assertTrue((A.lower == R.lower).all())
            self.assertTrue((A.upper == R.upper).all())
            R = (A + B).antiadd(A)
            self.assertTrue((B.lower == R.lower).all())
            self.assertTrue((B.upper == R.upper).all())

    def test_shape(self):
        for i in xrange(100):
            shape = _random_shape()
            A = NpInterval(np.ones(shape), 100 * np.ones(shape))
            B = NpInterval(np.ones(shape) * 2, np.ones(shape) * 3)
            R = A.antiadd(B)
            self.assertEqual(R.shape, shape)


class TestDNorm(TestCase):
    def foo(self, x, c, a, b):
        return (a * (1 - 2 * b) * x ** 2 + c) / (a * x ** 2 + c) ** (b + 1)
    def foo2(self, x, y, c, a, b):
        return -2 * a * b * x * y * ((a * (x ** 2 + y ** 2) + c) ** (-b - 1))

    def test_case0(self):
        a = 1.
        b = 0.5
        k = 1.
        # local range = 0
        act = np.asarray([[[[2.]], [[3.]], [[5.]]]])
        der = np.asarray([[[[1.]], [[1.]], [[1.]]]])
        activation = NpInterval(act, 1 * act)
        derivative = NpInterval(-der, der)

        res = self._count_norm(act, der, k, a, b, 0)
        R = derivative.op_d_norm(activation, act.shape, 0, k, a, b)
        self.assertTrue(np.isclose(res, R.upper).all())
        self.assertTrue(np.isclose(-res, R.lower).all())

        # local range = 2
        derivative = NpInterval(1 * der, 1 * der)

        c = k
        for i in xrange(3):
            c += act[0][i][0][0]**2
        res = np.zeros(act.shape)
        for i, j in product(xrange(3), xrange(3)):
            x = act[0][i][0][0]
            y = act[0][j][0][0]
            if i == j:
                res[:, i, ::] += self.foo(x, c - x ** 2, a, b)
            else:
                res[:, j, ::] += self.foo2(x, y, c - x**2 - y**2, a, b)


        res2 = self._count_norm(act, der, k, a, b, 2)
        R = derivative.op_d_norm(activation, act.shape, 2, k, a, b)
        self.assertTrue(np.isclose(res, res2).all())
        self.assertTrue(np.isclose(res, R.upper).all())
        self.assertTrue(np.isclose(res, R.lower).all())

    def test_case1(self):
        a = 1.
        b = 0.5
        k = 1.
        act = np.asarray([[[[2.]], [[3.]], [[5.]]]])
        der = np.asarray([[[[1.]], [[1.]], [[1.]]]])
        activation = NpInterval(act, 1 * act)
        derivative = NpInterval(-der, der)

        def foo(x, c):
            return (a * (1 - 2 * b) * x ** 2 + c) / (a * x ** 2 + c) ** (b + 1)

        def foo2(x, y, c):
            return -2 * a * b * x * y * (
                (a * (x ** 2 + y ** 2) + c) ** (-b - 1))

        c = k
        for i in xrange(3):
            c += act[0][i][0][0] ** 2
        res = np.zeros(act.shape)

        for i, j in product(xrange(3), xrange(3)):
            x = act[0][i][0][0]
            y = act[0][j][0][0]
            if i == j:
                res[:, i, ::] += foo(x, c - x ** 2)
            else:
                res[:, i, ::] += foo2(x, y, c - x ** 2 - y ** 2)

        R = derivative.op_d_norm(activation, act.shape, 2, k, a, b)
        self.assertTrue((R.lower <= -abs(res)).all())
        self.assertTrue((abs(res) <= R.upper).all())

    def test_case2(self):
        a = 4.
        b = 3
        k = 0.8
        # local range = 0
        act = np.asarray([[[[2.]], [[3.]], [[5.]]]])
        der = np.asarray([[[[1.]], [[1.]], [[1.]]]])
        activation = NpInterval(act, 1 * act)
        derivative = NpInterval(-der, der)

        res = self.foo(act, k, a, b)
        R = derivative.op_d_norm(activation, act.shape, 0, k, a, b)

        self.assertTrue(np.isclose(-res, R.upper).all())
        self.assertTrue(np.isclose(res, R.lower).all())

        # local range = 2
        derivative = NpInterval(1 * der, 1 * der)

        c = k
        for i in xrange(3):
            c += act[0][i][0][0] ** 2
        res = np.zeros(act.shape)

        for i, j in product(xrange(3), xrange(3)):
            x = act[0][i][0][0]
            y = act[0][j][0][0]
            if i == j:
                res[:, i, ::] += self.foo(x, c - x ** 2, a, b)
            else:
                res[:, j, ::] += self.foo2(x, y, c - x ** 2 - y ** 2, a, b)

        res2 = self._count_norm(act, der, k, a, b, 2)
        R = derivative.op_d_norm(activation, act.shape, 2, k, a, b)
        self.assertTrue(np.isclose(res, res2).all())
        self.assertTrue(np.isclose(res, R.upper).all())
        self.assertTrue(np.isclose(res, R.lower).all())

    def test_case3(self):
        a = 1.
        b = 0.5
        k = 1.
        # local range = 0
        act = np.asarray([[[[2.]], [[3.]], [[5.]]]])
        der = np.asarray([[[[-3.]], [[2.]], [[7.]]]])
        activation = NpInterval(act, 1 * act)
        derivative = NpInterval(der, 1 * der)

        res = self._count_norm(act, der, k, a, b, 0)
        R = derivative.op_d_norm(activation, act.shape, 0, k, a, b)
        self.assertTrue(np.isclose(res, R.upper).all())
        self.assertTrue(np.isclose(res, R.lower).all())

        # local range = 2
        derivative = NpInterval(1 * der, 1 * der)

        c = k
        for i in xrange(3):
            c += act[0][i][0][0] ** 2
        res = np.zeros(act.shape)
        for i, j in product(xrange(3), xrange(3)):
            x = act[0][i][0][0]
            y = act[0][j][0][0]
            if i == j:
                res[:, i, ::] += self.foo(x, c - x ** 2, a, b) * der[:, i, ::]
            else:
                res[:, j, ::] += self.foo2(x, y, c - x ** 2 - y ** 2, a, b) \
                                 * der[:, i, ::]

        res2 = self._count_norm(act, der, k, a, b, 2)
        R = derivative.op_d_norm(activation, act.shape, 2, k, a, b)
        self.assertTrue(np.isclose(res, res2).all())
        self.assertTrue(np.isclose(res, R.upper).all())
        self.assertTrue(np.isclose(res, R.lower).all())

    def _count_norm(self, act, der, k, alpha, beta, local_range):
        res = np.zeros_like(act)
        b, ch, h, w = der.shape
        for at_b, at_ch, at_h, at_w in product(xrange(b), xrange(ch),
                                               xrange(h), xrange(w)):
            c = k
            y = act[at_b, at_ch, at_h, at_w]
            for i in xrange(-local_range, local_range + 1):
                if i != 0 and 0 <= (at_ch + i) < ch:
                    c += act[at_b, at_ch + i, at_h, at_w]**2

            res[at_b, at_ch, at_h, at_w] += self.foo(y, c, alpha, beta) * \
                                            der[at_b, at_ch, at_h, at_w]

            for i in xrange(-local_range, local_range+1):
                if i != 0 and 0 <= at_ch + i < ch:
                    x = act[at_b, at_ch + i, at_h, at_w]
                    c -= x**2
                    res[at_b, at_ch + i, at_h, at_w] += self.foo2(x, y, c,
                                                                  alpha,
                                                                  beta) \
                                                        * der[at_b, at_ch,
                                                              at_h, at_w]
                    c += x**2
        return res

    def test_correct(self):
        for _ in xrange(100):
            s = randrange(1, 10)
            shape = (1, s, 1, 1)
            local_range = randrange(1, 3)
            k = uniform(1, 1)
            a = uniform(1, 1)
            b = uniform(0.5, 0.5)
            activations = _random_npinterval(shape)
            derivatives = _random_npinterval(shape)
            R = derivatives.op_d_norm(activations, shape, local_range,
                                      k, a, b)
            for _ in xrange(100):
                act = _rand_from_npinterval(activations)
                der = _rand_from_npinterval(derivatives)
                res = self._count_norm(act, der, k, a, b, local_range)

                if not (R.lower <= res).all() or not (res <= R.upper).all():
                    print "FAILED for a, b, k, lr", a, b, k, local_range
                    print "activations", activations
                    print "chosen:"
                    print act
                    print "derivatives", derivatives
                    print "chosen"
                    print der
                    print "results", R
                    print "chosen"
                    print res

                self.assertTrue((R.lower <= res + 1e-5).all())
                self.assertTrue((res <= R.upper + 1e-5).all())

    def test_correct_flat(self):
        for _ in xrange(100):
            s = randrange(1, 10)
            shape = (1, s, 1, 1)
            local_range = randrange(1, 3)
            k = uniform(1, 1)
            a = uniform(1, 1)
            b = uniform(0.5, 0.5)
            activations = _random_npinterval(shape)
            activations.lower = activations.upper * 1
            derivatives = _random_npinterval(shape)
            derivatives.lower = derivatives.upper * 1
            R = derivatives.op_d_norm(activations, shape, local_range,
                                      k, a, b)
            act = _rand_from_npinterval(activations)
            der = _rand_from_npinterval(derivatives)
            res = self._count_norm(act, der, k, a, b, local_range)

            self.assertTrue(np.isclose(R.lower, res).all())
            self.assertTrue(np.isclose(res, R.upper).all())

    def test_shape(self):
        for _ in xrange(100):
            shape = _random_shape(4, limit=100)
            local_range = randrange(0, 3)
            k = uniform(0.1, 10)
            a = uniform(0.1, 10)
            b = uniform(0.1, 3)

            diff = min(local_range, (shape[1] - 1) / 2)
            der_shape = (shape[0],) + (shape[1] - 2 * diff,) + shape[2:]
            A = _random_npinterval(shape)
            D = _random_npinterval(der_shape)
            R = D.op_d_norm(A, shape, local_range, k, a, b)
            self.assertEquals(A.shape, R.shape)


if __name__ == '__main__':
    main(verbosity=2)