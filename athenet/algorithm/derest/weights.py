"""Functions counting cost of weights in view of their activation/derivative.
"""
import numpy
import theano

from athenet.algorithm.derest.network import DerestNetwork
from athenet.algorithm.numlike.interval import Interval
from athenet.algorithm.deleting import delete_weights_by_global_fraction
from athenet.algorithm.derest.utils import _change_order


def get_derest_indicators(network, input):
    n = DerestNetwork(network)
    n.count_activations(input)
    output_nr = network.layers[-1].output_shape
    n.count_derivatives(input.derest_output(output_nr), output_nr)
    return n.count_derest()


def derest(network, fraction, (min_value, max_value)=(0., 255.)):
    input_shape = _change_order(network.layers[0].input_shape)
    input = Interval(theano.shared(numpy.full(input_shape, min_value)), theano.shared(numpy.full(input_shape, max_value)))
    indicators = get_derest_indicators(network, input)
    delete_weights_by_global_fraction(network.layers, fraction, indicators)




