"""Functions counting cost of weights in view of their activation/derivative.
"""

from random import randint

from athenet.algorithm.deleting import delete_weights_by_global_fraction
from athenet.algorithm.derest.network import DerestNetwork
from athenet.algorithm.derest.utils import change_order
from athenet.algorithm.numlike.npinterval import NpInterval
from athenet.algorithm.utils import to_indicators

from athenet.utils.constants import TMP_DIR


def length(value):
    """
    Computes indicator from Indicator

    :param Numlike values: values to count indicator from
    :return: float
    """
    return value.upper - value.lower


def divide_by_max(data):
    a = data.abs().amax()
    return data / (a.upper + 1e-6)


def get_derest_indicators(network, input_=None, count_function=length,
                          max_batch_size=None,
                          normalize_activations=lambda x: x,
                          normalize_derivatives=divide_by_max):
    """
    Returns indicators of importance using derest algorithm

    :param Network network: network to work with
    :param input_: possible input for network
    :type input_: Numlike or None
    :param function count_function: function to use
    :param batch_size: size of batch in computing derivatives
    :type batch_size: int or None
    :param function normalize_activations: function to normalize activations
        between layers
    :param function normalize_derivatives: function to normalize derivatives
        between layers
    :return array of numpy.ndarrays:
    """
    if input_ is None:
        input_ = NpInterval.from_shape(
            change_order(network.layers[0].input_shape),
            neutral=False
        )

    random_id = randint(0, 10**6)
    network_folder = TMP_DIR + str(random_id)
    derest_network = DerestNetwork(
        network, network_folder, normalize_activations, normalize_derivatives)
    derest_network.count_activations(input_)

    output_nr = network.layers[-1].output_shape
    if max_batch_size is None:
        max_batch_size = output_nr
    output = input_.derest_output(output_nr)
    for i in xrange(0, output_nr, max_batch_size):
        print "BATCH:", i
        derest_network.count_derivatives(output[i:(i+max_batch_size)])

    results = derest_network.count_derest(count_function)
    derest_network.delete_folder()
    return to_indicators(results)


def derest(network, fraction, input_=None, *args, **kwargs):
    """
    Delete set percentage of weights from network,

    :param Network network: network to delete weights from
    :param float fraction: fraction of weights to be deleted
    :param Numlike or None input_: possible input of network
    """

    indicators = get_derest_indicators(network, input_,
                                       *args, **kwargs)
    delete_weights_by_global_fraction(network.weighted_layers,
                                      fraction, indicators)
