"""Functions counting cost of weights in view of their activation/derivative.
"""
import numpy
import theano

from athenet.algorithm.derest.network import DerestNetwork
from athenet.algorithm.numlike.interval import Interval
from athenet.network import Network
from athenet.layers import ConvolutionalLayer, FullyConnectedLayer, Softmax, ReLU, MaxPool


def run_derest_run():
    n = Network([
        ConvolutionalLayer(image_shape=(8, 8, 1), filter_shape=(5, 5, 3)),
        ReLU(),
 #       MaxPool(poolsize=(2, 2)),
 #       ConvolutionalLayer(filter_shape=(2, 2, 50)),
 #       ReLU(),
 #       MaxPool(poolsize=(2, 2)),
 #       FullyConnectedLayer(n_out=500),
 #       ReLU(),
        FullyConnectedLayer(n_out=10),
        Softmax(),
    ])
    n2 = DerestNetwork(n)

    inp = Interval(theano.shared(numpy.full((1, 8, 8), 0.)), theano.shared(numpy.full((1, 8, 8), 1.)))
    n2.count_activations(inp)

    r = numpy.array([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    out = Interval(theano.shared(r), theano.shared(r))
    n2.count_derivatives(out, 1)

    return n2.count_derest()


def get_derest_indicators(network, input):
    n = DerestNetwork(network)
    n.count_activations(input)
    output_nr = network.layers[-1].output_shape
    n.count_derivatives(input.derest_output(output_nr), output_nr)
    return n.count_derest()




