import numpy
import cv2

from athenet.layers.conv import ConvolutionalLayer


def get_noise_indicators(filter, (diameter, sigma_color, sigma_space)):
    filter_as_image = numpy.array(filter * 255, dtype=numpy.uint8)
    sharpened_filter = cv2.bilateralFilter(filter_as_image, diameter,
                                           sigma_color, sigma_space)

    return (numpy.array(sharpened_filter, dtype=numpy.float32) -
            numpy.array(filter_as_image, dtype=numpy.float32)) / 255.


def sharpen_filter(filter, min_noise_indicator, max_value,
                   bilateral_filter_args):
    for filter_2d in filter:
        noise_indicators = get_noise_indicators(filter_2d, bilateral_filter_args)
        filter_2d[(abs(noise_indicators) >= min_noise_indicator)
               & (abs(filter_2d) < max_value)] = 0


def sharpen_filters_in_layer(layer, (min_noise_indicator,
                                     max_value, bilateral_filter_args)):
    W = layer.W
    for filter in W:
        sharpen_filter(filter, min_noise_indicator, max_value,
                        bilateral_filter_args)
    layer.W = W


def sharpen_filters_in_network(network, (min_noise_indicator, max_value,
                                         bilater_filter_args)):
    for layer in network.weighted_layers:
        if isinstance(layer, ConvolutionalLayer):
            sharpen_filters_in_layer(layer, (min_noise_indicator,
                                             max_value, bilater_filter_args))
