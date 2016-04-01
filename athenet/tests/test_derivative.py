"""Testing athenet.algorithm.derest.derivative functions.
"""

import numpy as np
import theano
import theano.tensor as T
import unittest
from math import e
from nose.tools import assert_almost_equal as aae, \
    assert_greater as ag
from numpy.testing import assert_array_almost_equal as arae
from athenet.algorithm.numlike import Interval as Itv, Nplike
from athenet.algorithm.derest.derivative import d_relu

theano.config.exception_verbosity = 'high'

A = np.array


def npl(x):
    return Nplike(A(x))


class DerivativeTest(unittest.TestCase):

    def prepare(self):
        self.v = np.arange(24) + 3.0
        self.at_v = 0
        return self.s, self.v, self.make_arr

    def s(self):
        if self.at_v >= len(self.v):
            raise ValueError
        ret = self.v[self.at_v]
        self.at_v += 1
        return ret

    def make_arr(self, shp):
        sz = np.prod(shp)
        a = np.ndarray(sz)
        for i in range(sz):
            a[i] = self.s()
        return a.reshape(shp)


class FullyConnectedDerivativeTest(DerivativeTest):
    pass


class ConvolutionalDerivativeTest(DerivativeTest):
    pass


class PoolDerivativeTest(DerivativeTest):
    pass


class SoftmaxDerivativeTest(DerivativeTest):
    pass


class NormDerivativeTest(DerivativeTest):
    pass


class DropoutDerivativeTest(DerivativeTest):
    pass


class ReluDerivativeTest(DerivativeTest):

    def test_case1(self):
        shp = (4, 3, 2)
        act = np.zeros(shp)
        for n_in in range(4):
            for h in range(3):
                for w in range(2):
                    act[n_in, h, w] += 100 * n_in + 10 * h + w
        thact = theano.shared(act)
        iact = Itv(thact, thact)
        thdout = theano.shared(np.ones(shp))
        idout = Itv(thdout, thdout)
        idin = d_relu(iact, idout)
        l, u = idin.eval()
        aae(l[0, 0, 0], 0.0)
        aae(u[0, 0, 0], 1.0)
        aae(l[2, 1, 1], 1.0)
        aae(l[2, 2, 1], 1.0)
        aae(l[1, 0, 1], 1.0)
        aae(l[2, 1, 1], 1.0)
        aae(l[2, 2, 0], 1.0)
        aae(l[1, 0, 1], 1.0)

    def test_case2(self):
        actl = theano.shared(np.array([-2, -1, -1, 0, 0, 1]))
        actu = theano.shared(np.array([-1, 1, 0, 0, 1, 2]))
        doutl = theano.shared(np.array([2, 3, 4, 7, 11, 13]))
        doutu = theano.shared(np.array([3, 5, 7, 11, 13, 17]))
        iact = Itv(actl, actu)
        idout = Itv(doutl, doutu)
        idin = d_relu(iact, idout)
        l, u = idin.eval()
        rl = np.array([0, 0, 0, 0, 0, 13])
        ru = np.array([0, 5, 7, 11, 13, 17])
        arae(l, rl)
        arae(u, ru)

    def test_case3(self):
        actl = theano.shared(np.array([-2, -1, -1, 0, 0, 1]))
        actu = theano.shared(np.array([-1, 1, 0, 0, 1, 2]))
        doutl = theano.shared(np.array([-3, -5, -7, -11, -13, -17]))
        doutu = theano.shared(np.array([-2, -3, -5, -7, -11, -13]))
        iact = Itv(actl, actu)
        idout = Itv(doutl, doutu)
        idin = d_relu(iact, idout)
        l, u = idin.eval()
        rl = np.array([0, -5, -7, -11, -13, -17])
        ru = np.array([0, 0, 0, 0, 0, -13])
        arae(l, rl)
        arae(u, ru)

if __name__ == '__main__':
    unittest.main(verbosity=2, catchbreak=True)
