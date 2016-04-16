"""Functions counting cost of weights in view of their activation/derivative.
"""
import numpy
import theano

from athenet.algorithm.derest.network import DerestNetwork
from athenet.algorithm.numlike.interval import Interval
from athenet.network import Network
from athenet.layers import ConvolutionalLayer, FullyConnectedLayer, Softmax


def run_derest_run():
    n = Network([
        ConvolutionalLayer(image_shape=(8, 8, 1), filter_shape=(5, 4, 1)),
#        ReLU(),
#        MaxPool(poolsize=(2, 2)),
        FullyConnectedLayer(n_out=10),
        Softmax()
    ])
    n2 = DerestNetwork(n)

    inp = Interval(theano.shared(numpy.full((1, 8, 8), 0.)), theano.shared(numpy.full((1, 8, 8), 1.)))
    n2.count_activations(inp)

    r = numpy.array([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    out = Interval(theano.shared(r), theano.shared(r))
    print n2.count_derivatives(out, 1)

def get_derest_indicators(network, input):
    pass




