"""Function for running algorithm on network for multiple cases. """

import os
import cPickle as pickle
import gzip
import copy

from athenet.utils import save_data_to_pickle, load_data_from_pickle, \
    count_zeros
from athenet.utils.results import Results


def get_error_rate(network):
    if network.data_loader.test_data_available:
        error_rate = 1.0 - network.test_accuracy()
    elif network.data_loader.val_data_available:
        error_rate = 1.0 - network.val_accuracy()
    else:
        raise Exception('test data and valid data not in Network')


def run_test(network, algorithm, config):
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
                        from where they are initially loaded. Stores dictionary
                        {config: algorithm(neural_network, config)}.
    :param verbose: If True, then progress of tests is being printed.
    :return: Dictionary {config: algorithm(neural_network, config)}.
    """
    save = results_pkl is not None
    layers = neural_network.weighted_layers
    results = Results(error_rate=get_error_rate(neural_network),
                      weighted_layers=layers, weights=[100 for layer in layers],
                      file=results_pkl)
    config_l = filter(lambda config: config not in results, config_l)
    n_of_cases = len(config_l)
    n_of_cases_passed = 0
    if verbose:
        print 'Cases to run:', n_of_cases
    for config in config_l:
        results.add_new_test_result(config, run_test(copy.deepcopy(neural_network), algorithm, config), save)
        n_of_cases_passed += 1
        if verbose:
            print 'Cases passed:', n_of_cases_passed, '/', n_of_cases
    if verbose:
        print 'Algorithm run successfully'
    return results
