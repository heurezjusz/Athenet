from athenet.algorithm import sparsify_smallest_on_network, sharpen_filters, \
    sparsify_smallest_on_layers, simple_neuron_deleter,\
    simple_neuron_deleter2, derest
from athenet.algorithm import get_filters_indicators, get_smallest_indicators,\
    get_nearest_to_global_mean_indicators, get_random_indicators, \
    get_nearest_to_layers_mean_indicators, get_derest_indicators, \
    delete_weights_by_global_fraction, delete_weights_by_layer_fractions
from athenet.models import lenet, alexnet, googlenet
from athenet.data_loader import MNISTDataLoader, ImageNetDataLoader, \
    AlexNetImageNetDataLoader, GoogleNetImageNetDataLoader
from athenet.layers import FullyConnectedLayer, ConvolutionalLayer

from derest_params import get_derest_params

import os

"""
    dictionary form algorithm shortcut to function to be called
"""
algorithms = {
    "sender": simple_neuron_deleter,
    "sender2": simple_neuron_deleter2,
    "rat": sparsify_smallest_on_network,
    "rat2": sparsify_smallest_on_layers,
    "filters": sharpen_filters,
    "derest": derest
    }


indicators = {
    "smallest": get_smallest_indicators,
    "global_mean": get_nearest_to_global_mean_indicators,
    "layers_mean": get_nearest_to_layers_mean_indicators,
    "filters": get_filters_indicators,
    "derest": get_derest_indicators,
    "random": get_random_indicators
}

default_types_of_layers = {
    "smallest": "all",
    "global_mean": "all",
    "layers_mean": "all",
    "filters": "conv",
    "derest": "all",
    "random": "all"
}

deleting = {
    "global": delete_weights_by_global_fraction,
    "layers": delete_weights_by_layer_fractions
}


def choose_layers(network, type_, indicators_, from_=None):
    if from_ is None:
        from_ = network.weighted_layers

    def get_layers(network, type_, indicators_):
        if type_ == "default":
            return get_layers(network, default_types_of_layers[indicators_],
                              indicators_)
        if type_ == "all":
            return [True for _ in network.weighted_layers]
        if type_ == "conv":
            return [isinstance(layer, ConvolutionalLayer)
                    for layer in network.weighted_layers]
        if type_ == "fully-connected":
            return [isinstance(layer, FullyConnectedLayer)
                    for layer in network.weighted_layers]

    layers = get_layers(network, type_, indicators_)
    return [a for (a, b) in zip(from_, layers) if b]


def get_indicators(network, type_, indicators_, args):
    f = indicators[indicators_]
    if indicators_ == "derest":
        ind = f(network, **get_derest_params(args))
        return choose_layers(network, type_, indicators_, ind)
    else:
        return f(choose_layers(network, type_, indicators_))


def get_network(network_type, val_size=None):
    """
        Returns a athenet.network of given type.
        :network_type: is a name of the type given as a string.
        :val_size: size of validation dataset
    """
    if val_size <= 0:
        val_size = None
    if network_type == "lenet":
        net = lenet()
        net.data_loader = MNISTDataLoader()
        return net
    if network_type == "alexnet":
        net = alexnet()
        net.data_loader = AlexNetImageNetDataLoader(val_size=val_size)
        return net
    if network_type == "googlenet":
        net = googlenet()
        net.data_loader = GoogleNetImageNetDataLoader(val_size=val_size)
        return net
    raise NotImplementedError


def get_file_name(args):
    if args.file:
        return args.file

    file_args = []
    for k, v in args.__dict__.iteritems():
        if k in ("file", "plot", "log", "batch_size", "examples"):
            continue
        if v in ("default", "none"):
            continue
        file_args.append(k + "_" + str(v))

    folder = "results/"
    if not os.path.exists(folder):
        os.makedirs(folder)

    return folder + "_".join(file_args)


def ok():
    """
        prints a good looking "OK" on the stdout.
    """
    print "[ \033[32mOK\033[39m ]"
