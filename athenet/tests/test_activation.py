"""Testing athenet.sparsifying.derest.activation functions.
"""

import unittest
from nose.tools import assert_true, assert_is, assert_equal
import numpy as np
from numpy.testing import assert_array_equal as are, \
    assert_array_almost_equal as arae
import theano
from theano import function
import theano.tensor as T
from athenet.sparsifying.utils import Interval as I, Nplike
from athenet.sparsifying.derest.activation import *

theano.config.exception_verbosity = 'high'

A = np.array


def npl(x):
    return Nplike(A(x))


def sprepare(self):
    self.v = np.arange(24) + 3.0
    self.at_v = 0
    return self.s, self.v, self.make_arr


def ss(self):
    if self.at_v >= len(self.v):
        raise ValueError
    ret = self.v[self.at_v]
    self.at_v += 1
    return ret


def smake_arr(self, shp):
    sz = np.prod(shp)
    a = np.ndarray(sz)
    for i in range(sz):
        a[i] = self.s()
    return a.reshape(shp)


class FullyConnectedActivationTest(unittest.TestCase):

    prepare = sprepare
    s = ss
    make_arr = smake_arr

    def test_1D_simple(self):
        res = fully_connected(npl([1]), A([2]), A([0]))
        arae(res.eval(), A([2]))

    def test_2D_simple_used_1D_of_weights(self):
        s, v, m = self.prepare()
        inp, w, b = m(2), m(2), A([1.0])
        res = fully_connected(inp, w, b)
        arae(res, A([v[0] * v[2] + v[1] * v[3] + 1.0]))

    def test_2D_simple_used_2D_of_weights(self):
        s, v, m = self.prepare()
        inp = m(1)
        w = m((1, 2))
        b = m(2)
        arae(fully_connected(inp, w, b), A([v[0] * v[1] + v[3],
                v[0] * v[2] + v[4]]))

    def test_2D_simple(self):
        s, v, m = self.prepare()
        inp = m(2)
        w = m((2, 2))
        b = m(2)
        arae(fully_connected(inp, w, b), A([v[0] * v[2] + v[1] * v[4] + v[6],
                v[0] * v[3] + v[1] * v[5] + v[7]]))

    def test_2D_2(self):
        s, v, m = self.prepare()
        inp = m(4)
        w = m((4, 2))
        b = m(2)
        rl = v[0] * v[4] + v[1] * v[6] + v[2] * v[8] + v[3] * v[10] + v[12]
        ru = v[0] * v[5] + v[1] * v[7] + v[2] * v[9] + v[3] * v[11] + v[13]
        arae(fully_connected(inp, w, b), A([rl, ru]))

    def test_3D_using_intervals(self):
        s, v, m = self.prepare()
        inpl = m(2)
        inpu = m(2)
        w = m((2, 2))
        b = A([1, 3])
        crl = A([v[0] * v[4] + v[1] * v[6] + 1,
                 v[0] * v[5] + v[1] * v[7] + 3])
        cru = A([v[2] * v[4] + v[3] * v[6] + 1,
                 v[2] * v[5] + v[3] * v[7] + 3])
        tinpl, tinpu = T.dvectors('inpl', 'inpu')
        iinp = I(tinpl, tinpu)
        res = fully_connected(iinp, w, b)
        d = {tinpl: inpl, tinpu: inpu}
        (rl, ru) = res.eval(d)
        arae(rl, crl)
        arae(ru, cru)

    def test_3D_negative_using_intervals(self):
        s, v, m = self.prepare()
        inpl = A([v[0], v[1]])
        inpu = A([v[2], v[3]])
        w = A([[v[4], -v[5]], [v[6], v[7]]])
        b = A([1, 3])
        crl = A([v[0] * v[4] + v[1] * v[6] + 1,
                v[0] * v[5] + v[3] * v[7] + 3])
        cru = A([v[2] * v[4] + v[3] * v[6] + 1,
                v[2] * v[5] + v[1] * v[7] + 3])
        tinpl, tinpu = T.dvectors('inpl', 'inpu')
        iinp = I(tinpl, tinpu)
        res = fully_connected(iinp, w, b)
        d = {tinpl: inpl, tinpu: inpu}
        (rl, ru) = res.eval(d)
        arae(rl, crl)
        arae(ru, cru)

    def test_negative(self):
        inp = A([1, -1])
        w = A([[1, 1], [1, -1]])
        b = A([0, 0])
        res = fully_connected(inp, w, b)
        c = A([0, 2])
        arae(res, c)

class ConvolutionalActivationTest(unittest.TestCase):

    prepare = sprepare
    s = ss
    make_arr = smake_arr

    def test_trivial(self):
        inp = npl(A([[[1]]]))
        w = npl(A([[[[2]]]]))
        b = npl(A([3]))
        f_shp = (w.shape[0], w.shape[2], w.shape[3])
        res = conv(inp, inp.shape, w, f_shp, b)
        arae(res.eval(), A([[[5]]]))

    def test_1_channel_input_1_conv_feature(self):
        inp = npl(A([[[0, 0], [2, 3]]]))
        w = npl(A([[[[7, 5], [3, 2]]]]))
        b = npl(A([4]))
        f_shp = (w.shape[0], w.shape[2], w.shape[3])
        res = conv(inp, inp.shape, w, f_shp, b)
        arae(res.eval(), A([[[35]]]))

    def test_1_channel_input_1_conv_feature2(self):
        inp = npl(np.zeros((1, 3, 3)))
        inp[0, 2, 1] = npl(2)
        inp[0, 2, 2] = npl(3)
        w = npl([[[[7, 5], [3, 2]]]])
        b = npl(A([4]))
        f_shp = (w.shape[0], w.shape[2], w.shape[3])
        res = conv(inp, inp.shape, w, f_shp, b)
        arae(res.eval(), A([[[4, 4], [18, 35]]]))

    def test_use_all_dims(self):
        inp = npl(np.zeros((2, 4, 4)))
        inp[0:2, 1:3, 1:3] = npl(A([[[2, 3], [5, 7]],
                                   [[0.2, 0.3], [0.5, 0.7]]]))
        w = npl(np.zeros((2, 2, 3, 3)))
        w[0, 0, 0, 0] = npl(1.0)
        w[0, 0, 0, 2] = npl(2.0)
        w[0, 0, 2, 0] = npl(3.0)
        w[0, 0, 1, 1] = npl(4.0)
        w[1, 0, 0, 0] = npl(5.0)
        w[1, 0, 0, 2] = npl(6.0)
        w[1, 0, 2, 0] = npl(7.0)
        w[1, 0, 2, 2] = npl(8.0)
        w[0, 1, 1, 1] = npl(9.0)
        w[0, 1, 1, 2] = npl(10.0)
        w[0, 1, 2, 1] = npl(11.0)
        w[0, 1, 2, 2] = npl(12.0)
        w[1, 1, 0, 0] = npl(13.0)
        w[1, 1, 2, 0] = npl(14.0)
        w[1, 1, 0, 1] = npl(15.0)
        w[1, 1, 2, 1] = npl(16.0)
        w_flipped = w[:, :, ::-1, ::-1]
        b = npl(A([[[13]], [[23]]]))
        f_shp = (w.shape[0], w.shape[2], w.shape[3])
        res = conv(inp, inp.shape, w_flipped, f_shp, b)
        arae(res.eval(), A([[[39.7, 50.4], [50.5, 49.3]],
                            [[87.0, 76.2], [44.0, 40.1]]]))

    def test_padding(self):
        inp = npl(A([[[2, 3], [5, 7]],
                    [[0.2, 0.3], [0.5, 0.7]]]))
        w = npl(np.zeros((2, 2, 3, 3)))
        w[0, 0, 0, 0] = npl(1.0)
        w[0, 0, 0, 2] = npl(2.0)
        w[0, 0, 2, 0] = npl(3.0)
        w[0, 0, 1, 1] = npl(4.0)
        w[1, 0, 0, 0] = npl(5.0)
        w[1, 0, 0, 2] = npl(6.0)
        w[1, 0, 2, 0] = npl(7.0)
        w[1, 0, 2, 2] = npl(8.0)
        w[0, 1, 1, 1] = npl(9.0)
        w[0, 1, 1, 2] = npl(10.0)
        w[0, 1, 2, 1] = npl(11.0)
        w[0, 1, 2, 2] = npl(12.0)
        w[1, 1, 0, 0] = npl(13.0)
        w[1, 1, 2, 0] = npl(14.0)
        w[1, 1, 0, 1] = npl(15.0)
        w[1, 1, 2, 1] = npl(16.0)
        w_flipped = w[:, :, ::-1, ::-1]
        b = npl(A([[[13]], [[23]]]))
        f_shp = (w.shape[0], w.shape[2], w.shape[3])
        res = conv(inp, inp.shape, w_flipped, f_shp, b, padding=(1, 1))
        arae(res.eval(), A([[[39.7, 50.4], [50.5, 49.3]],
                            [[87.0, 76.2], [44.0, 40.1]]]))

    def test_stride(self):
        inp = npl([[[2, 3], [5, 7]]])
        w = npl([[[[1, 2], [3, 4]]]])
        w_flipped = w[:, :, ::-1, ::-1]
        f_shp = (w.shape[0], w.shape[2], w.shape[3])
        b = npl([[[0]]])
        res = conv(inp, inp.shape, w_flipped, f_shp, b, padding=(1, 1),
                   stride=(2, 2))
        arae(res.eval(), A([[[8, 9], [10, 7]]]))

#    def test_interval(self):
#        inpl = A()
#        inp = npl([[[2, 3], [5, 7]]])
#        w = npl([[[[1, 2], [3, 4]]]])
#        w_flipped = w[:, :, ::-1, ::-1]
#        f_shp = (w.shape[0], w.shape[2], w.shape[3])
#        b = npl([[[0]]])
#        res = conv(inp, inp.shape, w_flipped, f_shp, b, padding=(1, 1),
#                   stride=(2, 2))
#        arae(res.eval(), A([[[8, 9], [10, 7]]]))

class MaxPoolActivationTest(unittest.TestCase):

    prepare = sprepare
    s = ss
    make_arr = smake_arr

class AvgPoolActivationTest(unittest.TestCase):

    prepare = sprepare
    s = ss
    make_arr = smake_arr

class SoftmaxActivationTest(unittest.TestCase):

    prepare = sprepare
    s = ss
    make_arr = smake_arr

class LRNActivationTest(unittest.TestCase):

    prepare = sprepare
    s = ss
    make_arr = smake_arr

class DropoutActivationTest(unittest.TestCase):

    prepare = sprepare
    s = ss
    make_arr = smake_arr

    def test_2x2_matrix(self):
        s, v, m = self.prepare()
        a = A([[s(), s()], [s(), s()]])
        res = dropout(a, 0.8)
        arae(res, 0.2 * a)

    def test_2x2_matrix_interval(self):
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
