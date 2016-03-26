"""Testing athenet.algorithm.derest.activation functions.
"""

from math import e
import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal as arae
import theano
import theano.tensor as T
from athenet.algorithm.numlike import Interval as Itv, Nplike
from athenet.algorithm.derest.activation import *

theano.config.exception_verbosity = 'high'

A = np.array


def npl(x):
    return Nplike(A(x))


class ActivationTest(unittest.TestCase):

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


class FullyConnectedActivationTest(ActivationTest):

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
        iinp = Itv(tinpl, tinpu)
        res = fully_connected(iinp, w, b)
        d = {tinpl: inpl, tinpu: inpu}
        (rl, ru) = res.eval(d)
        arae(rl, crl)
        arae(ru, cru)

    def test_3D_negative_weights_using_intervals(self):
        s, v, m = self.prepare()
        inpl = A([[[v[0], v[1]]]])
        inpu = A([[[v[2], v[3]]]])
        w = A([[v[4], -v[5], v[6]], [v[7], v[8], v[9]]])
        b = A([1, 3, 5])
        crl = A([v[0] * v[4] + v[1] * v[7] + 1,
                 v[2] * -v[5] + v[1] * v[8] + 3,
                 v[0] * v[6] + v[1] * v[9] + 5])
        cru = A([v[2] * v[4] + v[3] * v[7] + 1,
                 v[0] * -v[5] + v[3] * v[8] + 3,
                 v[2] * v[6] + v[3] * v[9] + 5])
        tinpl, tinpu = T.tensor3s('inpl', 'inpu')
        iinp = Itv(tinpl, tinpu)
        res = fully_connected(iinp, w, b)
        d = {tinpl: inpl, tinpu: inpu}
        (rl, ru) = res.eval(d)
        arae(rl, crl)
        arae(ru, cru)

    def test_negative(self):
        inp = npl([1, -1])
        w = A([[1, 1], [1, -1]])
        b = A([0, 0])
        res = fully_connected(inp, w, b)
        c = A([0, 2])
        arae(res.eval(), c)


class ConvolutionalActivationTest(ActivationTest):

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

    def test_interval_simple(self):
        inpl = A([[[-1, 3], [4, 7]]], dtype=theano.config.floatX)
        inpu = A([[[2, 3], [5, 9]]], dtype=theano.config.floatX)
        tinpl, tinpu = T.dtensor3s('tinpl', 'tinpu')
        iinp = Itv(tinpl, tinpu)
        w = A([[[[1, 2], [-3, 4]]]], dtype=theano.config.floatX)
        w_flipped = w[:, :, ::-1, ::-1]
        tw = theano.shared(w_flipped, borrow=True)
        b = A([[[0]]], dtype=theano.config.floatX)
        tb = theano.shared(b, borrow=True)
        inp_shape = (1, 2, 2)
        f_shp = (w.shape[0], w.shape[2], w.shape[3])
        res = conv(iinp, inp_shape, tw, f_shp, tb)
        d = {tinpl: inpl, tinpu: inpu}
        rl, ru = res.eval(d)
        arae(rl, A([[[18]]]))
        arae(ru, A([[[32]]]))

    def test_interval_3x3(self):
        inpl = A([[[-1, 3], [4, 7]]], dtype=theano.config.floatX)
        inpu = A([[[2, 3], [5, 9]]], dtype=theano.config.floatX)
        tinpl, tinpu = T.dtensor3s('tinpl', 'tinpu')
        iinp = Itv(tinpl, tinpu)
        w = A([[[[1, 2], [-3, 4]]]], dtype=theano.config.floatX)
        w_flipped = w[:, :, ::-1, ::-1]
        tw = theano.shared(w_flipped, borrow=True)
        b = A([0], dtype=theano.config.floatX)
        tb = theano.shared(b, borrow=True).dimshuffle(0, 'x', 'x')
        inp_shape = (1, 2, 2)
        f_shp = (w.shape[0], w.shape[2], w.shape[3])
        res = conv(iinp, inp_shape, tw, f_shp, tb, padding=(1, 1))
        d = {tinpl: inpl, tinpu: inpu}
        rl, ru = res.eval(d)
        arae(rl, A([[[-4, 6, -9], [14, 18, -24], [8, 18, 7]]]))
        arae(ru, A([[[8, 15, -9], [24, 32, -18], [10, 23, 9]]]))

    def test_group_1_in_2_out(self):
        inp = npl([[[2, 3], [5, 7]], [[2, 3], [5, 7]]])
        w = npl([[[[1, 2], [3, 4]]], [[[5, 6], [7, 8]]],
                 [[[5, 6], [7, 8]]], [[[1, 2], [3, 4]]]])
        w_flipped = w[:, :, ::-1, ::-1]
        f_shp = (w.shape[0], w.shape[2], w.shape[3])
        b = npl([[[0]]])
        res = conv(inp, inp.shape, w_flipped, f_shp, b, n_groups=2)
        arae(res.eval(), A([[[51.0]], [[119.0]], [[119.0]], [[51.0]]]))

    def test_group_2_in_1_out(self):
        inp = npl([[[1, 2], [3, 4]], [[5, 6], [7, 8]],
                   [[2, 3], [4, 5]], [[6, 7], [8, 9]]])
        w = npl([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                 [[[3, 4], [5, 6]], [[7, 8], [9, 1]]]])
        w_flipped = w[:, :, ::-1, ::-1]
        f_shp = (w.shape[0], w.shape[2], w.shape[3])
        b = npl([[[0]]])
        res = conv(inp, inp.shape, w_flipped, f_shp, b, n_groups=2)
        arae(res.eval(), A([[[204.0]], [[247.0]]]))


class PoolActivationTest(ActivationTest):

    def test_simple(self):
        inp = npl([[[1, 2], [3, 4]]])
        resmax = pool(inp, inp.shape, (1, 1), mode="max")
        resavg = pool(inp, inp.shape, (1, 1), mode="avg")
        arae(resmax.eval(), inp.eval())
        arae(resavg.eval(), inp.eval())

    def test_simple2(self):
        inp = Nplike(A([[[1, 2], [3, 4]]], dtype=float))
        resmax = pool(inp, inp.shape, (2, 2), mode="max")
        resavg = pool(inp, inp.shape, (2, 2), mode="avg")
        arae(resmax.eval(), A([[[4.0]]]))
        arae(resavg.eval(), A([[[2.5]]]))

    def test_2D1(self):
        inp = Nplike(A([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]))
        resmax = pool(inp, inp.shape, (2, 2), mode="max")
        resavg = pool(inp, inp.shape, (2, 2), mode="avg")
        arae(resmax.eval(), A([[[5.0, 6.0], [8.0, 9.0]]]))
        arae(resavg.eval(), A([[[3.0, 4.0], [6.0, 7.0]]]))

    def test_2D2(self):
        inp = Nplike(A([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]))
        resmax = pool(inp, inp.shape, (3, 3), mode="max")
        resavg = pool(inp, inp.shape, (3, 3), mode="avg")
        arae(resmax.eval(), A([[[9.0]]]))
        arae(resavg.eval(), A([[[5.0]]]))

    def test_3D(self):
        inp = Nplike(A([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                        [[2, 3, 4], [5, 6, 7], [8, 9, 1]]]))
        resmax = pool(inp, inp.shape, (2, 2), mode="max")
        resavg = pool(inp, inp.shape, (2, 2), mode="avg")
        arae(resmax.eval(), A([[[5.0, 6.0], [8.0, 9.0]],
                               [[6.0, 7.0], [9.0, 9.0]]]))
        arae(resavg.eval(), A([[[3.0, 4.0], [6.0, 7.0]],
                               [[4.0, 5.0], [7.0, 5.75]]]))

    def test_3D_interval(self):
        inpl = A([[[-1, 2, 3], [4, 5, 6], [7, -3, 0]],
                  [[2, 3, 4], [5, 6, 7], [8, 9, 1]]])
        inpu = A([[[1, 3, 4], [7, 5, 6], [7, 9, 9]],
                  [[2, 3, 4], [5, 6, 7], [8, 9, 1]]])
        tinpl, tinpu = T.tensor3s('tinpl', 'tinpu')
        iinp = Itv(tinpl, tinpu)
        resmax = pool(iinp, (2, 3, 3), (2, 2), mode="max")
        resavg = pool(iinp, (2, 3, 3), (2, 2), mode="avg")
        d = {tinpl: inpl, tinpu: inpu}
        rlmax, rumax = resmax.eval(d)
        rlavg, ruavg = resavg.eval(d)
        arae(rlmax, A([[[5, 6], [7, 6]], [[6, 7], [9, 9]]], dtype=float))
        arae(rumax, A([[[7, 6], [9, 9]], [[6, 7], [9, 9]]], dtype=float))
        arae(rlavg, A([[[10, 16], [13, 8]], [[16, 20], [28, 23]]],
                      dtype=float) / 4.0)
        arae(ruavg, A([[[16, 18], [28, 29]], [[16, 20], [28, 23]]],
                      dtype=float) / 4.0)


class SoftmaxActivationTest(ActivationTest):

    # TODO: Interval tests

    def test_simple(self):
        inp = npl([1, 2, 3])
        res = softmax(inp)
        s = e * (1 + e * (1 + e))
        arae(res.eval(), A([e / s, e ** 2 / s, e ** 3 / s]))

    def test_corner_cases(self):
        inps = [npl([1]), npl([2]), npl([1, 1]), npl([0]), npl([0, 0])]
        ress = [softmax(inp) for inp in inps]
        cress = [1, 1, 0.5, 1, 0.5]
        for (cres, res) in zip(cress, ress):
            arae(res.eval(), cres)


class LRNActivationTest(ActivationTest):

    pass


class DropoutActivationTest(ActivationTest):

    def test_2x2_matrix(self):
        s, v, m = self.prepare()
        a = npl([[[s(), s()], [s(), s()]]])
        res = dropout(a, 0.8)
        arae(res.eval(), a.eval() * A([0.2]))

    def test_2x2_matrix_interval(self):
        s, v, m = self.prepare()
        l = A([[[s(), s()], [s(), s()]]])
        u = A([[[s(), s()], [s(), s()]]])
        tl, tu = T.dtensor3s('l', 'u')
        i = Itv(tl, tu)
        drp = dropout(i, 0.8)
        d = {tl: l, tu: u}
        (rl, ru) = drp.eval(d)
        arae(rl, 0.2 * l)
        arae(ru, 0.2 * u)


class ReluActivationTest(ActivationTest):

    def test_simple(self):
        inp = npl([[[-3, -1, 1]]])
        arae(relu(inp).eval(), A([[[0, 0, 1]]]))

    def test_interval_simple(self):
        inpl = A([[[-3, -1, 1]]])
        inpu = A([[[-2, 3, 2]]])
        tinpl, tinpu = T.dtensor3s('tinpl', 'tinpu')
        iinp = Itv(tinpl, tinpu)
        res = relu(iinp)
        d = {tinpl: inpl, tinpu: inpu}
        rl, ru = res.eval(d)
        arae(rl, A([[[0, 0, 1]]]))
        arae(ru, A([[[0, 3, 2]]]))

    def test_interval_3D(self):
        inpl = A([[[-1, 2, -1], [0, 3, 5], [1, 2, 3]],
                  [[2, 3, 4], [-2, -3, -4], [-4, 0, 4]]])
        inpu = A([[[2, 2, 2], [1, 3, 5], [6, 5, 4]],
                  [[2, 3, 4], [-1, 0, 1], [4, 0, 4]]])
        tinpl, tinpu = T.dtensor3s('tinpl', 'tinpu')
        iinp = Itv(tinpl, tinpu)
        res = relu(iinp)
        d = {tinpl: inpl, tinpu: inpu}
        rl, ru = res.eval(d)
        arae(rl, A([[[0, 2, 0], [0, 3, 5], [1, 2, 3]],
                    [[2, 3, 4], [0, 0, 0], [0, 0, 4]]]))
        arae(ru, A([[[2, 2, 2], [1, 3, 5], [6, 5, 4]],
                    [[2, 3, 4], [0, 0, 1], [4, 0, 4]]]))

if __name__ == '__main__':
    unittest.main(verbosity=2, catchbreak=True)
