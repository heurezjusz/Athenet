#!/usr/bin/python2

import os
import cPickle as pickle
import gzip
import copy

from athenet.utils import save_data_to_pickle, load_data_from_pickle, zero_fraction


def run_algorithm(neural_network, algorithm, config_l, results_pkl=None,
                  verbose=False):
    """Runs algorithm on copy of neural_network for config_l cases.

    :neural_network: Instance of Network class to be copied and used for
                     algorithm.
    :algorithm: Function executing algorithm on network, takes Network and
                config parameters.
    :config_l:  List of configs to be passed to algorithm. For every config
                algorithm is being executed once.
    :results_pkl: File where results of algorithm are saved online and from
                  where they are initially loaded. Stores dictionary
                  {config: algorithm(neural_network, config)}.
    :verbose: If True, then progress of tests is being printed.
    :return: Dictionary {config: algorithm(neural_network, config)}.
    """
    results = {}
    if results_pkl:
        try:
            results = load_data_from_pickle(results_pkl)
        except:
            pass
    config_l = filter(lambda config: config not in results, config_l)
    n_of_cases = len(config_l)
    n_of_cases_passed = 0
    if verbose:
        print 'Cases to run:', n_of_cases
    for config in config_l:
        transformed_net = copy.deepcopy(neural_network)
        zeros_before = zero_fraction(transformed_net)
        algorithm(transformed_net, config)
        zeros_after = zero_fraction(transformed_net)
        zeroed_fraction = zeros_after - zeros_before
        error_rate = None
        if transformed_net.data_loader.test_data_available:
            error_rate = transformed_net.test_accuracy()
        elif transformed_net.data_loader.val_data_available:
            error_rate = transformed_net.val_accuracy()
        else:
            raise Exception('test data and valid data not in Network')
        results[config] = (zeroed_fraction, error_rate)
        n_of_cases_passed += 1
        if verbose:
            print 'Cases passed:', n_of_cases_passed, '/', n_of_cases
        if results_pkl:
            save_data_to_pickle(results, results_pkl)
    if verbose:
        print 'Algorithm run successfully'
    return results
