import numpy
import cv2

from athenet.layers.conv import ConvolutionalLayer


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
    sharpened_filter = cv2.bilateralFilter(filter_as_image,
                                           *bilateral_filter_args)

    return (numpy.array(sharpened_filter, dtype=numpy.float32) -
            numpy.array(filter_as_image, dtype=numpy.float32)) / 255.


def sharpen_filter(filter, min_noise_indicator, max_value,
                   bilateral_filter_args):
    """
    Delete weights in filter.

    This function, in given 3D filter,
    sets to zero weights which are smaller than max_value
    and which possibility of being a noise is greater than min_noise_value,

    :param numpy.ndarray filter: 3D filter
    :param float min_noise_indicator: minimal value of noise indicator
        enabling deleting the weight
    :param float max_value: values larger than max_value will be not deleted,
        regardless of the noise_indicator
    :param bilateral_filter_args: args for bilateral filtering
    """
    for filter_2d in filter:
        noise_indicators = get_noise_indicators(filter_2d,
                                                bilateral_filter_args)
        filter_2d[(abs(noise_indicators) >= min_noise_indicator)
                  & (abs(filter_2d) < max_value)] = 0


def sharpen_filters_in_layer(layer, (min_noise_indicator,
                                     max_value, bilateral_filter_args)):
    """
    Delete weights in layer.

    This function, in given layer,
    sets to zero filters' weights which are smaller than max_value
    and which possibility of being a noise is greater than min_noise_value,

    :param numpy.ndarray filter: 3D filter
    :param float min_noise_indicator: minimal value of noise indicator
        enabling deleting the weight
    :param float max_value: maximal value enabling deleting the weight
    :param bilateral_filter_args: args for bilateral filtering
    """
    W = layer.W
    for filter in W:
        sharpen_filter(filter, min_noise_indicator, max_value,
                       bilateral_filter_args)
    layer.W = W


def sharpen_filters_in_network(network, (min_noise_indicator, max_value,
                                         bilater_filter_args)):
    """
    Delete weights in network.

    This function, in given network's every convolutional layer,
    sets to zero weights which are smaller than max_value
    and which possibility of being a noise is greater than min_noise_value,

    :param numpy.ndarray filter: 3D filter
    :param float min_noise_indicator: minimal value of noise indicator
        enabling deleting the weight
    :param float max_value: maximal value enabling deleting the weight
    :param bilateral_filter_args: args for bilateral filtering
    """
    for layer in network.weighted_layers:
        if isinstance(layer, ConvolutionalLayer):
            sharpen_filters_in_layer(layer, (min_noise_indicator,
                                             max_value, bilater_filter_args))
