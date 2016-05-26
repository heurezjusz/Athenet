"""Stress testing athenet.algorithm.derest.derivative functions.
"""

import numpy as np
import theano
import theano.tensor as T
import unittest
from math import e
from athenet.algorithm.numlike import TheanoInterval
from athenet.algorithm.derest.derivative import *
from athenet.algorithm.derest.activation import *
from numpy.random import rand
import timeit

theano.config.exception_verbosity = 'high'
theano.config.optimizer = 'fast_compile'

A = np.array


def theano_var(x):
    return theano.shared(rand(*x).astype(theano.config.floatX))


def theano_interval(x):
    v = theano_var(x)
    return TheanoInterval(v, v)


class ActivationStressTest(unittest.TestCase):

    def check_time(self, name, start_time, constr_time, ex_time):
        print ''
        print name + ':'
        print 'constr_time:', constr_time - start_time
        print 'ex_time:', ex_time - constr_time

    def test_fully_connected(self):
        iinp = theano_interval((1024,))
        b = theano_interval((1000,))
        w = theano_var((1024, 1000))
        start_time = timeit.default_timer()
        iout = fully_connected(iinp, w, b)
        constr_time = timeit.default_timer()
        l, u = iout.eval()
        ex_time = timeit.default_timer()
        self.check_time('fully_connected', start_time, constr_time, ex_time)

    def test_convolutional(self):
        shp = (3, 224, 224)
        iinp = theano_interval(shp)
        w = theano_var((64, 3, 7, 7))
        b = theano_var((64,))
        start_time = timeit.default_timer()
        iout = conv(iinp, shp, w, (64, 7, 7),  b, stride=(2, 2),
                    padding=(3, 3))
        constr_time = timeit.default_timer()
        l, u = iout.eval()
        ex_time = timeit.default_timer()
        self.check_time('convolutional', start_time, constr_time, ex_time)

    def test_avg_pool(self):
        shp = (24, 16, 16)
        # TODO: test this (real case) (memory / time issues)
        # shp = (4, 192, 28, 28)
        iinp = theano_interval(shp)
        start_time = timeit.default_timer()
        iout = pool(iinp, shp, poolsize=(3, 3), stride=(1, 1), mode='avg')
        constr_time = timeit.default_timer()
        l, u = iout.eval()
        ex_time = timeit.default_timer()
        self.check_time('avg_pool', start_time, constr_time, ex_time)

    def test_max_pool(self):
        shp = (24, 16, 16)
        # TODO: test this (real case) (memory / time issues)
        # shp = (4, 192, 28, 28)
        iinp = theano_interval(shp)
        start_time = timeit.default_timer()
        iout = pool(iinp, shp, poolsize=(3, 3), stride=(1, 1), mode='max')
        constr_time = timeit.default_timer()
        l, u = iout.eval()
        ex_time = timeit.default_timer()
        self.check_time('max_pool', start_time, constr_time, ex_time)

    def test_softmax(self):
        # TODO: test this (real case) (memory / time issues)
        # shp = (1000,)
        # TODO: I think that softmax doesn't have to be calculated for Derest
        shp = (20,)
        iinp = theano_interval(shp)
        start_time = timeit.default_timer()
        iout = softmax(iinp, *shp)
        constr_time = timeit.default_timer()
        l, u = iout.eval()
        ex_time = timeit.default_timer()
        self.check_time('softmax', start_time, constr_time, ex_time)

    def test_norm(self):

        alpha = 0.00002
        beta = 0.75
        k = 1.0
        n = 5
        shp = (64, 56, 56)
        iinp = theano_interval(shp)
        start_time = timeit.default_timer()
        iout = norm(iinp, shp)
        constr_time = timeit.default_timer()
        l, u = iout.eval()
        ex_time = timeit.default_timer()
        self.check_time('norm', start_time, constr_time, ex_time)

    def test_dropout(self):
        iinp = theano_interval((50, 1024, 1, 1))
        start_time = timeit.default_timer()
        iout = d_dropout(iinp, 0.8)
        constr_time = timeit.default_timer()
        l, u = iout.eval()
        ex_time = timeit.default_timer()
        self.check_time('dropout', start_time, constr_time, ex_time)

    def test_relu(self):
        iinp = theano_interval((50, 1024, 1, 1))
        start_time = timeit.default_timer()
        iout = relu(iinp)
        constr_time = timeit.default_timer()
        l, u = iout.eval()
        ex_time = timeit.default_timer()
        self.check_time('relu', start_time, constr_time, ex_time)


class DerivativeStressTest(unittest.TestCase):

    def check_time(self, name, start_time, constr_time, ex_time):
        print ''
        print name + ':'
        print 'constr_time:', constr_time - start_time
        print 'ex_time:', ex_time - constr_time

    def test_fully_connected(self):
        idout = theano_interval((1, 1000))
        w = rand(1024, 1000)
        shp = (1, 1024)
        start_time = timeit.default_timer()
        din = d_fully_connected(idout, w, shp)
        constr_time = timeit.default_timer()
        l, u = din.eval()
        ex_time = timeit.default_timer()
        self.check_time('d_fully_connected', start_time, constr_time, ex_time)

    def test_convolutional(self):
        dout = theano_interval((1, 2, 14, 14))
        w = theano_var((2, 3, 7, 7))
        start_time = timeit.default_timer()
        din = d_conv(dout, (1, 3, 28, 28), (2, 7, 7), w, stride=(2, 2),
                     padding=(3, 3))
        # TODO: test this (real case) (memory / time issues)
        # dout = theano_interval((1, 64, 112, 112))
        # w = theano_var((64, 3, 7, 7))
        # start_time = timeit.default_timer()
        # din = d_conv(dout, (1, 3, 244, 244), (64, 7, 7), w, stride=(2, 2),
        #             padding=(3, 3))
        constr_time = timeit.default_timer()
        l, u = din.eval()
        ex_time = timeit.default_timer()
        self.check_time('d_convolutional', start_time, constr_time, ex_time)

    def test_avg_pool(self):
        shp = (4, 24, 14, 14)
        # TODO: test this (real case) (memory / time issues)
        # shp = (4, 192, 28, 28)
        iinp = theano_interval(shp)
        idout = theano_interval(shp)
        start_time = timeit.default_timer()
        din = d_pool(idout, iinp, shp, poolsize=(3, 3), padding=(1, 1),
                     stride=(1, 1), mode='avg')
        constr_time = timeit.default_timer()
        l, u = din.eval()
        ex_time = timeit.default_timer()
        self.check_time('d_avg_pool', start_time, constr_time, ex_time)

    def test_max_pool(self):
        shp = (2, 12, 3, 3)
        # TODO: test this (real case) (memory / time issues)
        # shp = (4, 192, 28, 28)
        iinp = theano_interval(shp)
        idout = theano_interval(shp)
        start_time = timeit.default_timer()
        din = d_pool(idout, iinp, shp, poolsize=(3, 3), padding=(1, 1),
                     stride=(1, 1), mode='max')
        constr_time = timeit.default_timer()
        l, u = din.eval()
        ex_time = timeit.default_timer()
        self.check_time('d_max_pool', start_time, constr_time, ex_time)

    def test_softmax(self):
        dout = TheanoInterval.derest_output(1000)
        start_time = timeit.default_timer()
        din = d_softmax(dout)
        constr_time = timeit.default_timer()
        l, u = din.eval()
        ex_time = timeit.default_timer()
        self.check_time('d_softmax', start_time, constr_time, ex_time)

    def test_norm(self):

        alpha = 0.00002
        beta = 0.75
        k = 1.0
        n = 5
        # TODO: Check higher batch size (memory issues)
        # iinp = theano_interval((50, 64, 56, 56))
        # idout = theano_interval((50, 64, 56, 56))
        iinp = theano_interval((10, 64, 56, 56))
        idout = theano_interval((10, 64, 56, 56))
        shp = (10, 64, 56, 56)
        start_time = timeit.default_timer()
        din = d_norm(idout, iinp, shp, n, k, alpha, beta)
        constr_time = timeit.default_timer()
        l, u = din.eval()
        ex_time = timeit.default_timer()
        self.check_time('d_norm', start_time, constr_time, ex_time)

    def test_dropout(self):
        idout = theano_interval((50, 1024, 1, 1))
        start_time = timeit.default_timer()
        idin = d_dropout(idout, 0.8)
        constr_time = timeit.default_timer()
        l, u = idin.eval()
        ex_time = timeit.default_timer()
        self.check_time('d_dropout', start_time, constr_time, ex_time)

    def test_relu(self):
        idout = theano_interval((50, 1024, 1, 1))
        iinp = theano_interval((50, 1024, 1, 1))
        start_time = timeit.default_timer()
        idin = d_relu(idout, iinp)
        constr_time = timeit.default_timer()
        l, u = idin.eval()
        ex_time = timeit.default_timer()
        self.check_time('d_relu', start_time, constr_time, ex_time)

if __name__ == '__main__':
    unittest.main(verbosity=2, catchbreak=True)
