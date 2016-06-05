"""Function for running algorithm on network for multiple cases. """

import datetime

from athenet.utils import count_zeros
from athenet.utils.results import Results


def get_error_rate(network):
    """
    Returns error rate in given network

    :param Network network:network to check
    :return float: error rate in network
    """
    if network.data_loader.test_data_available:
        error_rate = 1.0 - network.test_accuracy()
    elif network.data_loader.val_data_available:
        error_rate = 1.0 - network.val_accuracy()
    else:
        raise Exception('test data and validation data not in Network')
    return error_rate


def run_test(network, algorithm, config):
    """
    Runs algorithm in given network and return number of zeros and error rate
     in this network after running an algorithm

    :param Network network: network
    :param function algorithm: algorithm changing network
    :param float or iterable config: parameters for algorithm
    :return tuple(int, float): number of zeros and error rate
    """
    try:
        algorithm(network, *config)
    except TypeError:
        algorithm(network, config)

    zeros = count_zeros(network)
    return zeros, get_error_rate(network)


def run_algorithm(neural_network, algorithm, config_l, results_pkl=None,
                  verbose=False):
    """Runs algorithm on copy of neural_network for config_l cases.

    :param neural_network: Instance of Network class to be copied and used for
                           algorithm.
    :param algorithm: Function executing algorithm on network, takes Network
                      and config parameters.
    :param config_l: List of configs to be passed to algorithm. For every
                     config algorithm is being executed once.
    :param results_pkl: File where results of algorithm are saved online and
                        from where they are initially loaded.
    :param verbose: If True, then progress of tests is being printed.
    :return: Results
    """
    save = results_pkl is not None
    layers = neural_network.weighted_layers
    results = Results(
        error_rate=get_error_rate(neural_network),
        weighted_layers=[layer.__class__.__name__ for layer in layers],
        number_of_weights=[layer.W.size for layer in layers],
        file=results_pkl
    )
    config_l = results.get_new_test_configs(config_l)
    n_of_cases = len(config_l)
    n_of_cases_passed = 0
    weights = neural_network.get_params()
    if verbose:
        print datetime.datetime.now().strftime("%H:%M:%S.%f"), \
            'Cases to run:', n_of_cases
    for config in config_l:
        neural_network.set_params(weights)
        test_result = run_test(neural_network, algorithm, config)
        results.add_new_test_result(config, test_result, save)
        n_of_cases_passed += 1
        if verbose:
            print datetime.datetime.now().strftime("%H:%M:%S.%f"),\
                'Cases passed:', n_of_cases_passed, '/', n_of_cases
    if verbose:
        print datetime.datetime.now().strftime("%H:%M:%S.%f"),\
            'Algorithm run successfully'
    return results
