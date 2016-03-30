import numpy
import cv2

from athenet.layers.conv import ConvolutionalLayer
from athenet.algorithm.utils import set_zeros_by_global_fraction
from athenet.algorithm.sparsify_smallest import get_smallest_indicators


def get_noise_indicators(filter, bilateral_filter_args):
    """
    Returns possibility of being a noise .

    This function, for 2D filter with values in (-1, 1),
    computes possibility of being a noise for every value in filters,
    which is computed using bilateral filtering with given arguments

    :param numpy.ndarray filter: 2D filter
    :param bilateral_filter_args: args for bilateral filtering
    :type bilateral_filter_args: list or tuple
    """

    filter_as_image = numpy.array(filter * 255, dtype=numpy.uint8)
    sharpened_filter = cv2.bilateralFilter(filter_as_image, *bilateral_filter_args)

    return abs(numpy.array(sharpened_filter, dtype=numpy.float32) -
            numpy.array(filter_as_image, dtype=numpy.float32)) / 255.


def get_filters_indicators_in_conv_layer(layer, bilateral_filter_args):
    return numpy.array([[get_noise_indicators(f_2d, bilateral_filter_args) for f_2d in f] for f in layer.W])


def get_filters_indicators(layers, bilateral_filter_args):
    return numpy.array(
        [get_filters_indicators_in_conv_layer(layer, bilateral_filter_args) for layer in layers
         if isinstance(layer, ConvolutionalLayer)])


def sharpen_filters(network, (fraction, bilateral_filter_args)):
    """
    Delete weights in network.

    This function, in given network's every convolutional layer,
    sets to zero weights which are smaller than max_value
    and which possibility of being a noise is greater than min_noise_value,

    :param numpy.ndarray filter: 3D filter
    :param float fraction: fraction of weights to be changes to zeros
    :param bilateral_filter_args: args for bilateral filtering
    """
    conv_layers = [layer for layer in network.weighted_layers if isinstance(layer, ConvolutionalLayer)]
    filter_indicators = get_filters_indicators(conv_layers, bilateral_filter_args)
    smallest_indicators = get_smallest_indicators(conv_layers)
    set_zeros_by_global_fraction(conv_layers, fraction, filter_indicators * smallest_indicators)