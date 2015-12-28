""" Tests """

import theano
import theano.tensor as T
import numpy as np
from unittest import TestCase, main

from athenet import Network
from athenet.layers import ConvolutionalLayer, Softmax, FullyConnectedLayer, \
        ReLU, Activation


class TestNetworkBasics(TestCase):
    def test_layers_lists(self):
        def foo(x):
            return T.minimum(x, 0.)

        net = Network([
            ConvolutionalLayer(image_shape=(42, 21, 2), filter_shape=(5, 5, 3)),
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
        layer = ConvolutionalLayer(image_shape=(10, 10, 1), filter_shape=(5, 5, 3))
        out = eval_matrix_on_layer(layer, image)
        self.assertEqual(out.shape, (1, 3, 6, 6))

        layer2 = ConvolutionalLayer(filter_shape=(2, 3, 6), image_shape=(6, 6, 3))
        out = eval_tensor_on_layer(layer2, out)
        self.assertEqual(out.shape, (1, 6, 5, 4))

    def test_activation_layer(self):
        in_data = np.zeros((1, 4, 5, 6))
        for i in xrange(in_data.shape[1]):
            for j in xrange(in_data.shape[2]):
                for k in xrange(in_data.shape[3]):
                    in_data[0][i][j][k] = i + j ** 2 - 3 * k ** 3

        def foo(x):
            return x ** 3 / 16.

        layer = Activation(foo)
        out = eval_tensor_on_layer(layer, in_data)

        for i in xrange(in_data.shape[1]):
            for j in xrange(in_data.shape[2]):
                for k in xrange(in_data.shape[3]):
                    self.assertEqual(out[0][i][j][k], in_data[0][i][j][k] ** 3 / 16.)

    def test_relu_layer(self):
        in_data = np.zeros((1, 13, 5, 7))
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



if __name__ == '__main__':
    main()