import unittest
import theano
import theano.tensor as T
import numpy as np
from athenet.layers import InceptionLayer, ConvolutionalLayer
from athenet.algorithm.derest.layers import get_derest_layer, DerestLayer,\
    DerestInceptionLayer, DerestConvolutionalLayer
from athenet.algorithm.numlike import Interval
from athenet.algorithm.derest.utils import change_order, add_tuples,\
    make_iterable
from theano.gof.fg import MissingInputError


def eval_tensor_on_layers(layer1, layer2, tensor):
    layer1.input = T.tensor4()
    layer2.input_layer = layer1
    evaluate = theano.function(
        [layer1.input],
        layer2.output
    )
    return evaluate(tensor)


def eval_tensor_on_layer(layer, tensor):
    layer.input = T.tensor4('tmp')
    evaluate = theano.function(
        [layer.input],
        layer.output
    )
    return evaluate(tensor)


class TestActivationAndDerivativesShapes(unittest.TestCase):
    def _test_shapes(self, layer, batches=1):
        self.assertIsInstance(layer, DerestLayer)

        # activations
        inpl = T.tensor3('test_input_lower', dtype=theano.config.floatX)
        inpu = T.tensor3('test_input_upper', dtype=theano.config.floatX)
        input_in_theory = Interval(inpl, inpu)
        activation_in_theory = layer.count_activation(input_in_theory)
        layer.activations = input_in_theory
        act_input_shape = change_order(make_iterable(layer.layer.input_shape))

        activations = activation_in_theory.eval({
            inpl: np.ones(act_input_shape),
            inpu: np.ones(act_input_shape) * 2
        })[0]

        self.assertEquals(activations.shape,
                          change_order(layer.layer.output_shape))

        # derivatives
        outl = T.tensor4('test_output_lower', dtype=theano.config.floatX)
        outu = T.tensor4('test_output_upper', dtype=theano.config.floatX)
        output_in_theory = Interval(outl, outu)

        input_shape = add_tuples(batches,
                                 change_order(layer.layer.input_shape))
        output_shape = add_tuples(batches,
                                  change_order(layer.layer.output_shape))

        der_in_theory = layer.count_derivatives(output_in_theory, input_shape)
        try:
            derivatives = der_in_theory.eval({
                outl: np.ones(output_shape) * 3,
                outu: np.ones(output_shape) * 4
            })[0]
        except MissingInputError:
            derivatives = der_in_theory.eval({
                inpl: np.ones(act_input_shape),
                inpu: np.ones(act_input_shape) * 2,
                outl: np.ones(output_shape) * 3,
                outu: np.ones(output_shape) * 4
            })[0]

        self.assertEquals(derivatives.shape, input_shape)


class TestInceptionLayer(TestActivationAndDerivativesShapes):
    def prepare_inception_layer(self, image_size, n_filters):
        dummy_layer = ConvolutionalLayer((1, 1, 1), image_size)
        layer = InceptionLayer(n_filters=n_filters)
        input = np.ones((1,) + change_order(image_size))
        eval_tensor_on_layers(dummy_layer, layer, input)
        return layer

    def _run_test(self, image_size, n_filters, batches):
        layer = self.prepare_inception_layer(image_size, n_filters)
        derest_layer = get_derest_layer(layer)
        self.assertIsInstance(derest_layer, DerestInceptionLayer)
        self._test_shapes(derest_layer, batches)

    def test_case_1(self):
        self._run_test((3, 3, 1), [1, 2, 3, 4, 5, 6], 1)

    def test_simple(self):
        self._run_test((3, 3, 1), [1, 1, 1, 1, 1, 1], 1)

    def test_case_2(self):
        self._run_test((3, 4, 2), [2, 4, 3, 7, 3, 4], 2)


class TestConvolutionalLayer(TestActivationAndDerivativesShapes):
    def _run_test(self, filter_shape, image_shape, batches, padding=(0, 0),
                  stride=(1, 1), n_groups=1):
        layer = ConvolutionalLayer(filter_shape=filter_shape,
                                   image_shape=image_shape,
                                   padding=padding, stride=stride,
                                   n_groups=n_groups)

        input_shape = add_tuples((1,), change_order(image_shape))
        input = np.ones(input_shape)
        layer.input = input

        eval_tensor_on_layer(layer, input)
        derest_layer = get_derest_layer(layer)
        self.assertIsInstance(derest_layer, DerestConvolutionalLayer)
        self._test_shapes(derest_layer, batches)

    def test_simple(self):
        self._run_test((1, 1, 1), (10, 10, 1), 1)

    def test_square(self):
        self._run_test((3, 3, 1), (10, 10, 1), 3)

    def test_rectangle(self):
        self._run_test((5, 2, 4), (14, 8, 3), 7)

#    def test_paddig(self):
#        self._run_test((2, 3, 5), (12, 6, 8), 3, padding=(1, 1))

    def test_stride(self):
        self._run_test((2, 3, 5), (10, 10, 2), 2, stride=(2, 3))

    #def test_n_groups(self):
    #    self._run_test((2, 2, 3), (10, 5, 2), 2, n_groups=2)


if __name__ == '__main__':
    unittest.main(verbosity=2, catchbreak=True)
