"""Testing athenet.sparsifying.derest.activation functions.
"""

import unittest
from nose.tools import assert_true, assert_is, assert_equal
import numpy as np
from numpy.testing import assert_array_equal as are, \
    assert_array_almost_equal as arae
from theano import function
import theano.tensor as T
from athenet.sparsifying.utils.interval import Interval as I
from athenet.sparsifying.derest.activation import *

A = np.array

def prepare(self):
    self.v = np.arange(24) + 3.0
    self.at_v = 0
    return self.s, self.v, self.make_arr

def s(self):
    if self.at_v >= len(self.v):
        raise TypeError
    ret = self.v[self.at_v]
    self.at_v += 1
    return ret

def make_arr(self, shp):
    a = np.array(np.prod(shp))
    a.resize(shp)
    return a

class FullyConnectedActivationTest(unittest.TestCase):

    prepare = prepare
    s = s
    make_arr = make_arr

    def test1DSimple(self):
        s, v, m = self.prepare()
        #arae(fully_connected(1, 2, 0), A([2]))

    def test2DSimpleUsed1DOfWeights(self):
        s, v, m = self.prepare()
        inp, w, b = m(2), m(2), 1.0
        #arae(fully_connected(inp, w, b), v[0] * v[2] + v[1] * v[3] + 1.0)

    def test2DSimpleUSed2DOfWeights(self):
        s, v, m = self.prepare()
        inp = m(2)
        w = m((1, 2))
        b = A([1.0])
        #arae(fully_connected(inp, w, b), A([v[0] * v[1] + v[3],
        #        v[0] * v[2] + v[4]])

    def test2DSimple(self):
        s, v, m = self.prepare()
        inp = m(2)
        w = m((2, 2))
        b = m(2)
        #arae(fully_connected(inp, w, b), A(v[0] * v[2] + v[1] * v[4] + v[6],
        #        v[0] * v[3] + v[1] * v[5] + v[7])

    def test3D(self):
        s, v, m = self.prepare()
        inp = m(4)
        w = m((2, 2, 2))
        b = m(2)
        rl = v[0] * v[4] + v[1] * v[6] + v[2] * v[8] + v[3] * v[10] + v[12]
        ru = v[0] * v[5] + v[1] * v[7] + v[2] * v[9] + v[3] * v[11] + v[13]
        #arae(fully_connected(inp, w, b), A(rl, ru)

    def test3DUsingIntervals(self):
        s, v, m = self.prepare()
        inpl = m(4)
        wl = m((2, 2, 2))
        inpu = m(4)
        wu = m((2, 2, 2))
        bl = A([1, 3])
        bu = A([2, 4])
        crl = A([v[0] * v[4] + v[1] * v[6] + v[2] * v[8] + v[3] * v[10] + 1,
                v[0] * v[5] + v[1] * v[7] + v[2] * v[9] + v[3] * v[11] + 3])
        cru = A([v[12] * v[16] + v[13] * v[18] + v[14] * v[20] + \
                v[15] * v[22] + 2,
                v[12] * v[17] + v[13] * v[19] + v[14] * v[21] + \
                v[15] * v[23] + 4])
        tinpl, tinpu, tbl, tbu = T.dvectors('inpl', 'inpu', 'tbl', 'tbu')
        wl, wu = T.tensor3s('wl', 'wu')
        iinp = I(tinpl, tinpu)
        iw = I(twl, twu)
        ib = I(tbl, tbu)
        res = fully_connected(iinp, iw, ib)
        d = {tinpl: inpl, tinpu: inpu, tbl: bl, tbu: bu, twl: wl, twu: wu}
        (rl, ru) = res.eval(d)
        #arae(rl, crl)
        #arae(ru, cru)

class ConvolutionalActivationTest(unittest.TestCase):

    prepare = prepare
    s = s
    make_arr = make_arr

class MaxPoolActivationTest(unittest.TestCase):

    prepare = prepare
    s = s
    make_arr = make_arr

class SoftmaxActivationTest(unittest.TestCase):

    prepare = prepare
    s = s
    make_arr = make_arr

class LRNActivationTest(unittest.TestCase):

    prepare = prepare
    s = s
    make_arr = make_arr

class DropoutActivationTest(unittest.TestCase):

    prepare = prepare
    s = s
    make_arr = make_arr

    def test2x2Matrix(self):
        s, v, m = self.prepare()
        a = A([[s(), s()], [s(), s()]])
        res = dropout(a, 0.8)
        arae(res, 0.2 * a)

    def test2x2MatrixInterval(self):
        s, v, m = self.prepare()
        l = A([[s(), s()], [s(), s()]])
        u = A([[s(), s()], [s(), s()]])
        tl, tu = T.dmatrices('l', 'u')
        i = I(tl, tu)
        drp = dropout(i, 0.8)
        d = {tl: l, tu: u}
        (rl, ru) = drp.eval(d)
        arae(rl, 0.2 * l)
        arae(ru, 0.2 * u)

if __name__ == '__main__':
    unittest.main(verbosity=2, catchbreak=True)
