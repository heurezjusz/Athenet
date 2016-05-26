"""Functions counting cost of weights in view of their activation/derivative.
"""

from athenet.algorithm.deleting import delete_weights_by_global_fraction
from athenet.algorithm.derest.network import DerestNetwork
from athenet.algorithm.derest.utils import change_order
from athenet.algorithm.numlike.npinterval import NpInterval
from athenet.algorithm.utils import to_indicators


def sum_max(values):
    """
    Computes indicator from Numlike values

    :param Numlike values: values to count indicator from
    :return: float
    """
    return values.sum().upper


def get_derest_indicators(network, input_=None, count_function=sum_max,
                          max_batch_size=None, normalize_activations=False,
                          normalize_derivatives=True):
    """
    Returns indicators of importance using derest algorithm

    :param Network network: network to work with
    :param Numlike or None input_: possible input for network
    :param function count_function: function to use
    :param int or None max_batch_size: size of batch in computing derivatives
    :param bool normalize_activations: whenever to normalize activations
        between layers
    :param bool normalize_derivatives: whenever to normalize derivatives
        between layers
    :return array of integers:
    """
    if input_ is None:
        input_ = NpInterval.from_shape(
            change_order(network.layers[0].input_shape),
            neutral=False
        )

    n = DerestNetwork(network)
    n.count_activations(input_, normalize_activations)

    output_nr = network.layers[-1].output_shape
    if max_batch_size is None:
        max_batch_size = output_nr
    output= input_.derest_output(output_nr)
    for i in xrange(0, output_nr, max_batch_size):
        n.count_derivatives(output[i:(i+max_batch_size)],
                            normalize_derivatives)

    results = n.count_derest(count_function)
    return to_indicators(results)


def derest(network, fraction, input_=None, *args, **kwargs):
    """
    Delete set percentage of weights from network,

    :param Network network: network to delete weights from
    :param float fraction: fraction of weights to be deleted
    :param Numlike or None input_: possible input of network
    """

    input_shape = change_order(network.layers[0].input_shape)
    indicators = get_derest_indicators(network, input_, sum_max,
                                       *args, **kwargs)
    delete_weights_by_global_fraction(network.weighted_layers,
                                      fraction, indicators)
