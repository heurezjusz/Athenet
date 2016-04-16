"""Functions counting cost of weights in view of their activation/derivative.
"""

from athenet.algorithm.derest.network import DerestNetwork
from athenet.algorithm.numlike.numlike import Numlike


def get_derest_indicators(network, input):
    n = DerestNetwork(network)
    n.count_activations(input)
    for outp in input.derest_output(10):
        n.count_derivatives(outp)


