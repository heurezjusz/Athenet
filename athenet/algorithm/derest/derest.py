"""Functions counting cost of weights in view of their activation/derivative.
"""
import numpy
import theano

from athenet.algorithm.deleting import delete_weights_by_global_fraction
from athenet.algorithm.derest.network import DerestNetwork
from athenet.algorithm.derest.utils import _change_order
from athenet.algorithm.numlike.interval import Interval
from athenet.algorithm.utils import to_indicators


def sum_max(a, b, many=False):
    c = numpy.amax(numpy.abs(b), 0)
    if many:
        return a + numpy.sum(c)
    else:
        return a + c


def get_derest_indicators(network, input, count_function):
    n = DerestNetwork(network)
    n.count_activations(input)
    output_nr = network.layers[-1].output_shape
    n.count_derivatives(input.derest_output(output_nr))
    results = n.count_derest(count_function)
    return to_indicators(results)


def derest(network, fraction, (min_value, max_value)=(0., 255.)):
    input_shape = _change_order(network.layers[0].input_shape)
    input = Interval(
        theano.shared(numpy.full(input_shape, min_value)),
        theano.shared(numpy.full(input_shape, max_value))
    )
    indicators = get_derest_indicators(network, input, sum_max)
    delete_weights_by_global_fraction(network.weighted_layers, fraction, indicators)
