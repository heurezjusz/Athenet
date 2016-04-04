import numpy
import cv2

from athenet.algorithm.deleting import delete_weights_by_layer_fractions
from athenet.algorithm.sparsify_smallest import get_smallest_indicators


def get_bilateral_noise_indicators(filter, bilateral_filter_args):
    """
    Returns possibility of being a noise .

    This function, for 2D filter with values in (-1, 1),
    computes possibility of being a noise for every value in filters,
    which is computed using bilateral filtering with given arguments

    :param numpy.ndarray filter: 2D filter
    :param bilateral_filter_args: args for bilateral filtering
    :type bilateral_filter_args: list or tuple
    """
    assert len(filter.shape) == 2

    filter_as_image = numpy.array(filter * 255, dtype=numpy.uint8)
    sharpened_filter = cv2.bilateralFilter(filter_as_image,
                                           *bilateral_filter_args)

    return abs(numpy.array(sharpened_filter, dtype=numpy.float32) -
               numpy.array(filter_as_image, dtype=numpy.float32)) / 255.


def get_filters_indicators_in_conv_layer(layer, bilateral_filter_args):
    """
    Return indicators of being a noise for convolutional layer.

    This function, for convolutional layer,
    return indicators of being a noise
    computed using bilateral filtering with given arguments
    for every 2d filter

    :param layer: convolutional layer
    :param bilateral_filter_args: args for bilateral filtering
    :type bilateral_filter_args: list or tuple
    :return numpy.ndarray
    """
    assert len(layer.W.shape) == 4

    def filter_indicator(filter_3d):
        return [
            get_bilateral_noise_indicators(filter_2d, bilateral_filter_args)
            for filter_2d in filter_3d]
    return numpy.array([filter_indicator(filter_3d) for filter_3d in layer.W])


def get_filters_indicators(layers, bilateral_filter_args):
    """
    Returns indicators of being a noise for layers.

    This function, for every layer,
    returns indicatos of being a noise
    computed using bilateral filtering with given arguments
    for every 2d filter in every layer

    :param layers:
    :type layers: list or numpy.array or tuple
    :param bilateral_filter_args:args for bilateral filtering
    :return: numpy.ndarray
    """

    return numpy.array(
        [get_filters_indicators_in_conv_layer(layer, bilateral_filter_args)
         for layer in layers
         if len(layer.W.shape) == 4])  # only for convolutional layers


def sharpen_filters(network, (fraction, bilateral_filter_args)):
    """
    Delete weights in network.

    This function, in given network's every convolutional layer,
    sets to zero weights which are smaller than max_value
    and which possibility of being a noise is greater than min_noise_value,

    :param Network network: network for sparsifying
    :param tuple args: args for filter algorithm
    """

    conv_layers = [layer for layer in network.weighted_layers
                   if len(layer.W.shape) == 4]  # only for convolutional layers
    filter_indicators = get_filters_indicators(conv_layers,
                                               bilateral_filter_args)
    smallest_indicators = get_smallest_indicators(conv_layers)
    delete_weights_by_layer_fractions(conv_layers, fraction,
                                      filter_indicators * smallest_indicators)
