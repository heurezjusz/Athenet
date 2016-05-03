from athenet.algorithm.numlike import NpInterval
from unittest import TestCase, main
from random import randrange
from itertools import product
import numpy as np


def _random_shape():
    result = None
    limit = 10 ** 4
    size = 1
    for i in xrange(randrange(1, 7)):
        l = randrange(1, 10)
        if result is None:
            result = (l,)
        else:
            result += (l,)
        size *= l
        if size >= limit:
            return result
    return result


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


class Just(TestCase):
    def test(self):
        shape = (2, 5, 3, 3)
        act = NpInterval(np.ones(shape), np.ones(shape) * 2)
        norm = act.op_d_norm(act, shape, 5, 1, 1, 0.5)


if __name__ == '__main__':
    main(verbosity=2)