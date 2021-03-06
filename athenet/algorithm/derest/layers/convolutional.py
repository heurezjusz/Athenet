from itertools import product
import theano
import numpy

from athenet.algorithm.derest.layers.layer import DerestLayer
from athenet.algorithm.derest.utils import change_order
from athenet.algorithm.numlike import assert_numlike


class DerestConvolutionalLayer(DerestLayer):
    need_activation = True
    need_derivatives = True

    def __init__(self, *args):
        super(DerestConvolutionalLayer, self).__init__(*args)
        self.theano_ops = {}

    def _count_activation(self, layer_input):
        """
        Return estimated activations

        :param Numlike layer_input: input for layer
        :return Numlike:
        """
        return a_conv(
            layer_input, change_order(self.layer.input_shape),
            self.layer.W, change_order(self.layer.filter_shape),
            theano.shared(self.layer.b), self.layer.stride, self.layer.padding,
            self.layer.n_groups
        )

    def _count_derivatives(self, layer_output, input_shape):
        """
        Returns estimated impact of input of layer on output of
        network.

        :param Numlike layer_output: impact of input of next layer
            on output of network
        :param tuple input_shape:
        :return Numlike:
        """
        return d_conv(
            layer_output, input_shape,
            change_order(self.layer.filter_shape), self.layer.W,
            self.layer.stride, self.layer.padding, self.layer.n_groups,
            self.theano_ops
        )

    def _get_activation_for_weight(self, activation, i2, i3):
        """
        For given weight returns activations for inputs used by this weight.

        :param Numlike activation: activation with padded edges
        :param integer i2: weight's index
        :param integer i3: weight's index
        :return Numlike: activations for weight
        """

        n2, n3, _ = self.layer.input_shape
        m2, m3, _ = self.layer.filter_shape
        p2, p3 = self.layer.padding
        s2, s3 = self.layer.stride

        l2 = n2 + 2 * p2 - m2 + i2 + 1
        l3 = n3 + 2 * p3 - m3 + i3 + 1

        return activation[:, i2:l2:s2, i3:l3:s3]

    def _count_derest_for_weight(self, act, der, W, j0, j1):
        act = self._get_activation_for_weight(act, j0, j1)

        a1, a2, a3 = act.shape
        d1, d2, d3 = der.shape
        final_shape = (d1, a1, a2, a3)

        act = act.reshape((1, a1, a2, a3)).broadcast(final_shape)
        der = der.reshape((d1, 1, d2, d3)).broadcast(final_shape)

        inf = (der * act).sum((2, 3))
        inf = inf.reshape((W.shape[0], W.shape[1]))
        return inf * W[:, :, j0, j1]

    def count_derest(self, count_function):
        """
        Returns indicators of each weight importance

        :param function count_function: function to count indicators,
            takes Numlike and returns float
        :return list of numpy arrays:
        """
        indicators = numpy.zeros_like(self.layer.W)
        W = self.layer.W

        derivatives = self.load_derivatives()

        input_shape = (1, ) + change_order(self.layer.input_shape)
        activation = self.load_activations().reshape(input_shape)
        activation = activation.\
            reshape_for_padding(input_shape, self.layer.padding)
        activation = activation.reshape(activation.shape[1:])

        act_group_size = activation.shape[0] / self.layer.n_groups
        der_group_size = derivatives.shape[0] / self.layer.n_groups
        w_group_size = W.shape[0] / self.layer.n_groups

        for n_group in xrange(self.layer.n_groups):
            act_first = n_group * act_group_size
            act = \
                activation[act_first:(act_first + act_group_size), :, :]
            der_first = n_group * der_group_size
            der = \
                derivatives[der_first:(der_first + der_group_size), :, :]
            w_first = n_group * w_group_size
            weights = W[w_first:(w_first + w_group_size), :, :, :]

            for j2, j3 in product(xrange(W.shape[2]), xrange(W.shape[3])):
                ind = count_function(self._count_derest_for_weight(
                    act, der, weights, j2, j3))
                indicators[w_first:(w_first + w_group_size), :, j2, j3] = ind

        return [indicators]


def a_conv(layer_input, image_shape, weights, filter_shape, biases,
           stride=(1, 1), padding=(0, 0), n_groups=1):
    """Returns estimated activation of convolutional layer.

    :param layer_input: input Numlike in input_shp format
                (number of input channels, image height, image width)
    :param image_shape: image shape in the format (number of input channels,
                                                   image height,
                                                   image width)
    :param weights: Weights tensor in format (number of output channels,
                                              number of input channels,
                                              filter height,
                                              filter width)
    :param filter_shape: filter shape in the format (number of output channels,
                                                     filter height,
                                                     filter width)
    :param biases: biases in convolution
    :param stride: pair representing interval at which to apply the filters.
    :param padding: pair representing number of zero-valued pixels to add on
                    each side of the input.
    :param n_groups: number of groups input and output channels will be split
                     into, two channels are connected only if they belong to
                     the same group.
    :type layer_input: Numlike or numpy.ndarray or theano tensor
    :type image_shape: tuple of 3 integers
    :type weights: numpy.ndarray or theano tensor
    :type filter_shape: tuple of 3 integers
    :type biases: 1D np.array or theano.tensor
    :type stride: pair of integers
    :type padding: pair of integers
    :type n_groups: integer
    :rtype: Numlike
    """
    assert_numlike(layer_input)
    try:
        return layer_input.op_conv(weights, image_shape, filter_shape, biases,
                                   stride, padding, n_groups)
    except NotImplementedError:
        # n_in, h, w - number of input channels, image height, image width
        n_in, h, w = image_shape
        # n_out, fh, fw - number of output channels, filter height, filter
        # width
        n_out, fh, fw = filter_shape
        # g_in - number of input channels per group
        g_in = n_in / n_groups
        # g_out - number of output channels per group
        g_out = n_out / n_groups
        pad_h, pad_w = padding
        stride_h, stride_w = stride
        # see: flipping kernel
        flipped_weights = weights[:, :, ::-1, ::-1]
        input_type = type(layer_input)
        padded_input_shape = (n_in, h + 2 * pad_h, w + 2 * pad_w)
        padded_input = input_type.from_shape(padded_input_shape, neutral=True)
        padded_input[:, pad_h:(pad_h + h), pad_w:(pad_w + w)] = \
            layer_input
        # setting new n_in, h, w for padded input, now you can forget about
        # padding
        n_in, h, w = padded_input_shape
        output_h = (h - fh) / stride_h + 1
        output_w = (w - fw) / stride_w + 1
        output_shp = (n_out, output_h, output_w)
        result = input_type.from_shape(output_shp, neutral=True)
        for at_g in xrange(0, n_groups):
            # beginning and end of at_g'th group of input channel in input
            at_in_from = at_g * g_in
            at_in_to = at_in_from + g_in
            # beginning and end of at_g'th group of output channel in weights
            at_out_from = at_g * g_out
            at_out_to = at_out_from + g_out
            for at_h in xrange(0, h - fh + 1, stride_h):
                # at_out_h - output position in height dimension corresponding
                # to filter at position at_h
                at_out_h = at_h / stride_h
                for at_w in xrange(0, w - fw + 1, stride_w):
                    # at_out_w - output position in width dimension
                    # corresponding to filter at position at_w
                    at_out_w = at_w / stride_w
                    # input slice that impacts on (at_out_h, at_out_w) in
                    # output
                    input_slice = padded_input[at_in_from:at_in_to,
                                               at_h:(at_h + fh),
                                               at_w:(at_w + fw)]
                    # weights slice that impacts on (at_out_h, at_out_w) in
                    # output
                    weights_slice = flipped_weights[at_out_from:at_out_to, :,
                                                    :, :]
                    conv_sum = input_slice * weights_slice
                    conv_sum = conv_sum.sum(axis=(1, 2, 3), keepdims=False)
                    result[at_out_from:at_out_to, at_out_h, at_out_w] = \
                        conv_sum
        result = result + biases
        return result


def d_conv(output, input_shape, filter_shape, weights,
           stride=(1, 1), padding=(0, 0), n_groups=1, theano_ops=None):
    """Returns estimated impact of input of convolutional layer on output of
    network.

    :param Numlike output: estimated impact of output of layer on output
                           of network in shape (batch_size, number of channels,
                           height, width)
    :param input_shape: shape of layer input in the format
                        (number of batches,
                        number of input channels,
                        image height,
                        image width)
    :type input_shape: tuple of 4 integers
    :param filter_shape: filter shape in the format (number of output channels,
                                                     filter height,
                                                     filter width)
    :type filter_shape: tuple of 3 integers
    :param weights: Weights tensor in format (number of output channels,
                                              number of input channels,
                                              filter height,
                                              filter width)
    :type weights: numpy.ndarray or theano tensor
    :param stride: pair representing interval at which to apply the filters.
    :type stride: pair of integers
    :param padding: pair representing number of zero-valued pixels to add on
                    each side of the input.
    :type padding: pair of integers
    :param n_groups: number of groups input and output channels will be split
                     into, two channels are connected only if they belong to
                     the same group.
    :type n_groups: integer
    :param theano_ops: map in which theano graph might be saved
    :type theano_ops: map of theano functions
    :returns: Estimated impact of input on output of network
    :rtype: Numlike
    """
    res = output.op_d_conv(input_shape, filter_shape,
                           weights, stride, padding, n_groups, theano_ops)
    return res
