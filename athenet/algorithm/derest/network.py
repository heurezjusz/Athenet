import theano
import numpy

from athenet.layers import ConvolutionalLayer, FullyConnectedLayer, InceptionLayer
from athenet.algorithm.derest.activation import count_activation
from athenet.algorithm.derest.derivative import count_derivative
from athenet.algorithm.derest.utils import _change_order
from itertools import product


def _add_tuples(a, b):
    if not isinstance(a, tuple):
        a = (a, )
    if not isinstance(b, tuple):
        b = (b, )
    return a + b


class DerestNetwork():

    def __init__(self, network):
        self.network = network
        self.layers = [self._get_derest_layer(layer)
                       for layer in network.layers]

    def _get_derest_layer(self, layer):
        if isinstance(layer, FullyConnectedLayer):
            return DerestFullyConnectedLayer(layer)
        elif isinstance(layer, ConvolutionalLayer):
            return DerestConvolutionalLayer(layer)
        elif isinstance(layer, InceptionLayer):
            return DerestInceptionLayer(layer)
        else:
            return DerestLayer(layer)

    def _get_layer_input_shape(self, i):
        if i > 0:
            return self.layers[i - 1].layer.output_shape
        return self.layers[i].layer.input_shape

    def count_activations(self, inp):
        for layer in self.layers:
            inp = layer.count_activation(inp)
        return inp

    def count_derivatives(self, outp):
        batches = outp.shape.eval()[0]
        for i in range(len(self.layers) - 1, -1, -1):
            input_shape = _add_tuples(
                batches,
                _change_order(self._get_layer_input_shape(i))
            )
            outp = self.layers[i].count_derivatives(
                outp,
                input_shape
            )
        return outp

    def count_derest(self, f):
        result = []
        for layer in self.layers:
            indices = layer.count_derest(f)
            if indices is not None:
                result.append(indices)
        return result


class DerestLayer():

    def __init__(self, layer):
        self.layer = layer
        self.activations = None
        self.derivatives = None

    def count_activation(self, input):
        self.activations = input
        return count_activation(input, self.layer)

    def count_derivatives(self, output, input_shape):
        self.derivatives = output
        return count_derivative(output, self.activations,
                                input_shape, self.layer)

    def count_derest(self, f):
        pass


class DerestFullyConnectedLayer(DerestLayer):

    def count_derest(self, count_function):
        indicators = numpy.zeros_like(self.layer.W)
        nr_of_batches = self.derivatives.shape.eval()[0]
        for i in range(nr_of_batches):
            act = self.activations.reshape((self.layer.input_shape, 1))
            der = self.derivatives[i].reshape((1, self.layer.output_shape))
#            indicators = indicators + numpy.amax((act.dot(der) * self.layer.W).eval(), 0)
            b =  (act.dot(der) * self.layer.W).eval()
            indicators = count_function(indicators,b)
        return indicators


class DerestConvolutionalLayer(DerestLayer):

    def _get_activation_for_weight(self, i1, i2, i3):
        #no padding or strides yet considered
        n1, n2, _ = self.layer.input_shape
        m1, m2, _ = self.layer.filter_shape
        return self.activations[i1, i2:(n1-m2+i2+1), i3:(n2-m2+i3+1)]

    def count_derest(self, f):
        indicators = numpy.zeros_like(self.layer.W)

        i0, i1, i2, i3 = self.layer.W.shape
        for batch_nr in range(self.derivatives.shape.eval()[0]): #for every batch
            der = self.derivatives[batch_nr]
            for j1, j2, j3, j4 in product(range(i0), range(i1), range(i2), range(i3)):
                y = self._get_activation_for_weight(j2, j3, j4)
                x = (der[j1] * y * self.layer.W[j1, j2, j3, j4]).eval()
                indicators[j1, j2, j3, j4] = f(indicators[j1, j2, j3, j4], x, True)

        return indicators


class DerestInceptionLayer(DerestLayer):

    def count_derest(self, f):
        raise NotImplementedError

