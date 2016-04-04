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
from athenet.algorithm.derest.derivative import *

theano.config.exception_verbosity = 'high'

A = np.array


def npl(x):
    return Nplike(A(x))


def thv(x):
    return theano.shared(np.array(x, dtype=float))


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

    def test_1D_simple(self):
        dout = thv([[1]])
        idout = Itv(dout, dout)
        w = thv([[2]])
        shp = (1,)
        din = d_fully_connected(idout, w, shp)
        l, u = din.eval()
        arae(l, u)
        arae(l, A([[2]]))

    def test_2D_simple_used_1D_of_weights(self):
        dout = thv([[3, 6]])
        idout = Itv(dout, dout)
        w = thv([9, 12])
        shp = (1,)
        din = d_fully_connected(idout, w, shp)
        l, u = din.eval()
        arae(l, u)
        arae(l, A([[99]]))

    def test_2D_simple_used_2D_of_weights(self):
        dout = thv([[3, 0]])
        idout = Itv(dout, dout)
        w = thv([[6, 0], [9, 0]])
        shp = (2,)
        din = d_fully_connected(idout, w, shp)
        l, u = din.eval()
        arae(l, u)
        arae(l, A([[18, 27]]))

    def test_2D_1(self):
        dout = thv([[3, 6]])
        idout = Itv(dout, dout)
        w = thv([[9, 15], [12, 18]])
        shp = (2,)
        din = d_fully_connected(idout, w, shp)
        l, u = din.eval()
        arae(l, u)
        arae(l, A([[117, 144]]))

    def test_2D_Intervals(self):
        doutl = thv([[-3, -6, 3]])
        doutu = thv([[9, -3, 6]])
        idout = Itv(doutl, doutu)
        w = thv([[2, -3, -3], [-3, 1, 2], [5, -4, 3], [-2, -3, -4]])
        shp = (2, 2)
        din = d_fully_connected(idout, w, shp)
        l, u = din.eval()
        arae(l, A([[[-15, -27], [6, -33]]]))
        arae(u, A([[[27, 18], [87, 12]]]))

    def test_2D_batches(self):
        dout = thv([[3, 6], [1, 2]])
        idout = Itv(dout, dout)
        w = thv([[9, 15], [12, 18]])
        shp = (2,)
        din = d_fully_connected(idout, w, shp)
        l, u = din.eval()
        arae(l, u)
        arae(l, A([[117, 144], [39, 48]]))

    def test_2D_batches2(self):
        dout = A([[3, 6], [1, 2]])
        tdout = T.dmatrix('tdout')
        idout = Itv(tdout, tdout)
        d = {tdout: dout}
        w = thv([[9, 15], [12, 18]])
        shp = (2,)
        din = d_fully_connected(idout, w, shp)
        l, u = din.eval(d)
        arae(l, u)
        arae(l, A([[117, 144], [39, 48]]))


class ConvolutionalDerivativeTest(DerivativeTest):
    pass


class MaxPoolDerivativeTest(DerivativeTest):
    pass


class AvgPoolDerivativeTest(DerivativeTest):
    pass

class SoftmaxDerivativeTest(DerivativeTest):

    def test_case1(self):
        dout = Itv.derest_output(1)
        din = d_softmax(dout)
        l, u = din.eval()
        arae(l, u)
        arae(l, A([[1]]))

    def test_case2(self):
        dout = Itv.derest_output(3)
        din = d_softmax(dout)
        l, u = din.eval()
        arae(l, u)
        arae(l, A([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))


class NormDerivativeTest(DerivativeTest):
    pass


class DropoutDerivativeTest(DerivativeTest):

    def test_case1(self):
        doutl = thv([[-3, 0, 3], [3, -3, -5], [-3, -2, 1]])
        doutu = thv([[-3, 2, 3], [5, 3, 2], [-1, 3, 3]])
        idout = Itv(doutl, doutu)
        idin = d_dropout(idout, 0.8)
        l, u = idin.eval()
        rl = A([[-0.6, 0, 0.6], [0.6, -0.6, -1], [-0.6, -0.4, 0.2]])
        ru = A([[-0.6, 0.4, 0.6], [1, 0.6, 0.4], [-0.2, 0.6, 0.6]])
        arae(l, rl)
        arae(u, ru)


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
        idin = d_relu(idout, iact)
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
        actl = thv([-2, -1, -1, 0, 0, 1])
        actu = thv([-1, 1, 0, 0, 1, 2])
        doutl = thv([2, 3, 4, 7, 11, 13])
        doutu = thv([3, 5, 7, 11, 13, 17])
        iact = Itv(actl, actu)
        idout = Itv(doutl, doutu)
        idin = d_relu(idout, iact)
        l, u = idin.eval()
        rl = A([0, 0, 0, 0, 0, 13])
        ru = A([0, 5, 7, 11, 13, 17])
        arae(l, rl)
        arae(u, ru)

    def test_case3(self):
        actl = thv([-2, -1, -1, 0, 0, 1])
        actu = thv([-1, 1, 0, 0, 1, 2])
        doutl = thv([-3, -5, -7, -11, -13, -17])
        doutu = thv([-2, -3, -5, -7, -11, -13])
        iact = Itv(actl, actu)
        idout = Itv(doutl, doutu)
        idin = d_relu(idout, iact)
        l, u = idin.eval()
        rl = A([0, -5, -7, -11, -13, -17])
        ru = A([0, 0, 0, 0, 0, -13])
        arae(l, rl)
        arae(u, ru)

if __name__ == '__main__':
    unittest.main(verbosity=2, catchbreak=True)
