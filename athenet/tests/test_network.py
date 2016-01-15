""" Tests """

import theano
import theano.tensor as T
import numpy as np
from unittest import TestCase, main

from athenet import Network
from athenet.layers import ConvolutionalLayer, Softmax, FullyConnectedLayer, \
    ReLU, Activation, LRN, MaxPool, Dropout
from athenet.utils import DataLoader


class TestNetworkBasics(TestCase):
    def test_layers_lists(self):
        def foo(x):
            return T.minimum(x, 0.)

        net = Network([
            ConvolutionalLayer(image_shape=(42, 21, 2),
                               filter_shape=(5, 5, 3)),
            ReLU(),
            ConvolutionalLayer(filter_shape=(5, 5, 3)),
            Activation(foo),
            FullyConnectedLayer(10),
            ReLU(),
            Softmax(),
            FullyConnectedLayer(3),
        ])

        self.assertEqual(len(net.convolutional_layers), 2)
        self.assertEqual(len(net.weighted_layers), 4)
        self.assertEqual(len(net.layers), 8)


class DummyDataLoader(DataLoader):
    def __init__(self, data_in, data_out):
        """ data_in: list of batches witch input data
            data_out: list of batches witch output data
        """
        super(DummyDataLoader, self).__init__()
        self.data_in = theano.shared(np.asarray((data_in),
                                                dtype=theano.config.floatX))
        self.data_out = T.cast(theano.shared(np.asarray(data_out)), 'int32')

        self.train_set_size = len(data_in)
        self.val_set_size = len(data_in)
        self.test_set_size = len(data_in)

        self.train_data_available = True
        self.val_data_available = False
        self.test_data_available = True

    def train_input(self, batch_index):
        return self.data_in[batch_index:batch_index + 1]

    def train_output(self, batch_index):
        return self._get_subset(self.data_out, batch_index)[0]

    def test_input(self, batch_index):
        return self.train_input(batch_index)

    def test_output(self, batch_index):
        return self.train_output(batch_index)


class TestTrainingProcess(TestCase):
    def test_training_process(self):
        net = Network([FullyConnectedLayer(n_in=3, n_out=6),
                       ReLU(),
                       FullyConnectedLayer(n_out=2),
                       Softmax()])
        self.assertEqual(net.batch_size, 1)
        data_in = []
        data_out = []
        for i in xrange(2):
            for j in xrange(2):
                for k in xrange(2):
                    data_in.append([[[1. * i, 1. * j, 1. * k]]])
                    data_out.append([i ^ j ^ k])
        net.data_loader = DummyDataLoader(data_in, data_out)
        net.train(n_epochs=500, batch_size=1)

        net.batch_size = 3
        correct = 0.
        for i in xrange(2):
            for j in xrange(2):
                for k in xrange(2):
                    raw = net.evaluate([[[i, j, k]]])[0]
                    out = net.evaluate([[[i, j, k]]])[1]
                    self.assertTrue(raw[out[0]] > raw[out[1]])
                    if out[0] == i ^ j ^ k:
                        correct += 1.

        self.assertEqual(correct / len(data_in), net.test_accuracy())
        # it should be automatically corrected
        self.assertEqual(net.batch_size, 1)


def eval_tensor_on_layer(layer, tensor):
    layer.input = T.tensor4('tmp')
    evaluate = theano.function(
        [layer.input],
        layer.output
    )
    return evaluate(tensor)


def eval_matrix_on_layer(layer, matrix):
    height, width = matrix.shape
    tensor = np.resize(matrix, (1, 1, height, width))
    return eval_tensor_on_layer(layer, tensor)


class TestLayers(TestCase):
    def test_convolutional_layer(self):
        image = np.ndarray((10, 10), dtype=theano.config.floatX)
        layer = ConvolutionalLayer(image_shape=(10, 10, 1),
                                   filter_shape=(5, 5, 3))
        out = eval_matrix_on_layer(layer, image)
        self.assertEqual(out.shape, (1, 3, 6, 6))

        layer2 = ConvolutionalLayer(image_shape=(6, 6, 3),
                                    filter_shape=(2, 3, 6))
        out = eval_tensor_on_layer(layer2, out)
        self.assertEqual(out.shape, (1, 6, 5, 4))

    def test_activation_layer(self):
        in_data = np.zeros((1, 4, 5, 6), dtype=theano.config.floatX)
        for i in xrange(in_data.shape[1]):
            for j in xrange(in_data.shape[2]):
                for k in xrange(in_data.shape[3]):
                    in_data[0][i][j][k] = i + j ** 2 - 3 * k ** 3

        def foo(x):
            return x ** 2 / 16. - x

        layer = Activation(foo)
        out = eval_tensor_on_layer(layer, in_data)

        for i in xrange(in_data.shape[1]):
            for j in xrange(in_data.shape[2]):
                for k in xrange(in_data.shape[3]):
                    tmp = in_data[0][i][j][k]
                    self.assertEqual(out[0][i][j][k],
                                     tmp ** 2 / 16. - tmp)

    def test_relu_layer(self):
        in_data = np.zeros((1, 13, 5, 7), dtype=theano.config.floatX)
        for i in xrange(in_data.shape[1]):
            for j in xrange(in_data.shape[2]):
                for k in xrange(in_data.shape[3]):
                    in_data[0][i][j][k] = i + j ** 2 - 5 * k

        layer = ReLU()
        out = eval_tensor_on_layer(layer, in_data)

        for i in xrange(in_data.shape[1]):
            for j in xrange(in_data.shape[2]):
                for k in xrange(in_data.shape[3]):
                    if in_data[0][i][j][k] < 0.:
                        self.assertEqual(out[0][i][j][k], 0.)
                    else:
                        self.assertEqual(out[0][i][j][k], in_data[0][i][j][k])

    def test_lrn_layer(self):
        in_data = np.zeros((1, 13, 5, 7), dtype=theano.config.floatX)
        for i in xrange(in_data.shape[1]):
            for j in xrange(in_data.shape[2]):
                for k in xrange(in_data.shape[3]):
                    in_data[0][i][j][k] = i + j ** 2 - 5 * k

        layer = LRN(4)
        self.assertEqual(layer.local_range, 5)
        self.assertEqual(layer.k, 1)
        self.assertEqual(layer.alpha, 0.0002)
        self.assertEqual(layer.beta, 0.75)

        output = eval_tensor_on_layer(layer, in_data)

        for i in xrange(in_data.shape[1]):
            for j in xrange(in_data.shape[2]):
                for k in xrange(in_data.shape[3]):
                    self.assertTrue(abs(in_data[0][i][j][k]) >=
                                    abs(output[0][i][j][k]))

    def test_maxpool_layer(self):
        in_data = np.zeros((1, 2, 4, 4), dtype=theano.config.floatX)
        for i in xrange(4):
            for j in xrange(4):
                in_data[0][0][i][j] = i * 4 + j
        in_data[0][1][1][1] = 1.
        in_data[0][1][1][2] = 2.
        in_data[0][1][2][1] = 3.
        in_data[0][1][2][2] = 4.

        layer = MaxPool(poolsize=(2, 2), stride=(1, 1))
        output = eval_tensor_on_layer(layer, in_data)

        self.assertEqual(output.shape, (1, 2, 3, 3))
        for i in xrange(3):
            for j in xrange(3):
                self.assertEqual(in_data[0][0][i + 1][j + 1],
                                 output[0][0][i][j])
                self.assertEqual(output[0][1][i][j],
                                 1 + 2 * ((i + 1) / 2) + (j + 1) / 2)

        layer = MaxPool(poolsize=(2, 2))
        output = eval_tensor_on_layer(layer, in_data)

        self.assertEqual(output.shape, (1, 2, 2, 2))
        for i in xrange(2):
            for j in xrange(2):
                self.assertEqual(output[0][0][i][j], 5 + 8 * i + 2 * j)
                self.assertEqual(output[0][1][i][j], 1 + 2 * i + j)

    def test_softmax_layer(self):
        in_data = np.zeros((5, 6), dtype=theano.config.floatX)
        in_data[0][0] = -8
        for i in xrange(1, in_data.shape[0]):
            in_data[i][0] = in_data[i - 1][0] + 2
        for j in xrange(1, in_data.shape[1]):
            for i in xrange(in_data.shape[0]):
                in_data[i][j] = in_data[i][j - 1] + 2

        layer = Softmax()
        layer.input = T.matrix('tmp')
        foo = theano.function(
            [layer.input],
            layer.output
        )
        output = foo(in_data)

        for i in xrange(in_data.shape[0]):
            self.assertTrue(output[i][0] < 1)
            for j in xrange(1, in_data.shape[1]):
                self.assertTrue(output[i][j] < 1)
                self.assertTrue(output[i][j - 1] < output[i][j])

    def test_dropout_layer(self):
        in_data = np.zeros((1, 13, 15, 27), dtype=theano.config.floatX)
        for i in xrange(in_data.shape[1]):
            for j in xrange(in_data.shape[2]):
                for k in xrange(in_data.shape[3]):
                    in_data[0][i][j][k] = i + j ** 2 - 5 * k

        layer = Dropout()
        self.assertEqual(layer.p_dropout, 0.5)

        layer = Dropout(0.75)
        output = eval_tensor_on_layer(layer, in_data)

        for i in xrange(in_data.shape[1]):
            for j in xrange(in_data.shape[2]):
                for k in xrange(in_data.shape[3]):
                    self.assertEqual(output[0][i][j][k],
                                     in_data[0][i][j][k] / 4.)

        layer.train_input = T.tensor4('tmp')
        foo = theano.function(
            [layer.train_input],
            layer.train_output
        )
        output2 = foo(in_data)

        dropped_out = 0.
        for i in xrange(in_data.shape[1]):
            for j in xrange(in_data.shape[2]):
                for k in xrange(in_data.shape[3]):
                    if output2[0][i][j][k] != 0.:
                        self.assertEqual(output2[0][i][j][k],
                                         in_data[0][i][j][k])
                    else:
                        dropped_out += 1.
        self.assertTrue(abs(dropped_out / in_data.size - 0.75) <= 0.05)

    def test_fully_connected_layer(self):
        in_data = np.zeros((1, 4), dtype=theano.config.floatX)
        layer = FullyConnectedLayer(13, in_data.size)
        output = eval_tensor_on_layer(layer, in_data)
        self.assertEqual(output.size, 13)
        for i in xrange(output.size):
            self.assertEqual(output[0][i], 0.)


if __name__ == '__main__':
    main(verbosity=2, catchbreak=True)
