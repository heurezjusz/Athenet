"""For any neuron, we calculate its range of activation for inputs of neural
networks in given range. For simplicity, we assume it is [0, 255]. We do it
from the beginning to the end of the network. Functions at first try to use
"""

from athenet.algorithm.numlike import Numlike, assert_numlike

# TODO: All functions below will be implemented.


def conv(layer_input, input_shp, weights, filter_shp, biases, stride=(1, 1),
         padding=(0, 0), n_groups=1):
    """Returns estimated activation of convolutional layer.

    :param layer_input: Input Numlike
    :param input_shp: Shape of input in the format
                (number of input channels, image height, image width)
    :param weights: Weights tensor in format (number of output channels,
                                              number of input channels,
                                              filter height,
                                              filter width)
    :param filter_shp: Filter shape in the format (number of output channels,
                                                   filter height,
                                                   filter width)
    :param biases: Biases in convolution in shape (0, 'x', 'x'). e.g.
                   theano.shared(numpy.ndarray([0], dtype=theano.config.floatX),
                   borrow=True).dimshuffle(0, 'x', 'x')
    :param stride: Pair representing interval at which to apply the filters.
    :param padding: Pair representing number of zero-valued pixels to add on
                    each side of the input.
    :param n_groups: Number of groups input and output channels will be split
                     into. Two channels are connected only if they belong to
                     the same group.
    :type layer_input: Numlike or numpy.ndarray or theano tensor
    :type input_shp: integer tuple
    :type weights: numpy.ndarray or theano tensor
    :type filter_shp: integer tuple
    :type biases: 1D np.array or theano.tensor
    :type stride: integer pair
    :type padding: integer pair
    :type n_groups: integer
    :rtype: Numlike
    """
    assert_numlike(layer_input)
    # h, w, n_in - image height, image width, number of input channels
    n_in, h, w = input_shp
    # fw, fh, n_out - filter height, filter width, number of output channels
    n_out, fh, fw = filter_shp
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
    padded_input = input_type.from_shape(padded_input_shape)
    padded_input[0:n_in, pad_h:(pad_h + h), pad_w:(pad_w + w)] = \
        layer_input
    # setting new h, w, n_in for padded input, you can forget about padding
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
            # at_out_h - height of output corresponding to filter at
            # position at_h
            at_out_h = at_h / stride_h
            for at_w in xrange(0, w - fw + 1, stride_w):
                # at_out_w - height of output corresponding to filter at
                # position at_w
                at_out_w = at_w / stride_w
                # input slice that impacts on (at_out_h, at_out_w) in output
                input_slice = padded_input[at_in_from:at_in_to,
                                           at_h:(at_h + fh),
                                           at_w:(at_w + fw)]
                # weights slice that impacts on (at_out_h, at_out_w) in output
                weights_slice = flipped_weights[at_out_from:at_out_to, :, :, :]
                conv_sum = input_slice * weights_slice
                conv_sum = conv_sum.sum(axis=(1, 2, 3), keepdims=False)
                result[at_out_from:at_out_to, at_out_h, at_out_w] = conv_sum
    result = result + biases
    return result


def dropout(layer_input, p_dropout):
    """Returns estimated activation of dropout layer.

    :param Numlike layer_input: input Numlike
    :param float p_dropout: probability of dropping in dropout
    :rtype: Numlike
    """
    assert_numlike(layer_input)
    return layer_input * (1.0 - p_dropout)


def fully_connected(layer_input, weights, biases):
    """Returns estimated activation of fully connected layer.

    :param Numlike layer_input: input Numlike
    :param weights: weights of fully connected layer in order (n_in, n_out)
    :param biases: biases of fully connected layer of size n_out
    :type weights: 2D numpy.ndarray or theano.tensor
    :type biases: 1D numpy.ndarray or theano.tensor
    """
    assert_numlike(layer_input)
    flat_input = layer_input.flatten()
    try:
        return flat_input.dot(weights) + biases
    except NotImplementedError:
        return (flat_input * weights.T).sum(1) + biases


def norm(input_layer, local_range=5, k=1, alpha=0.0002, beta=0.75):
    """Returns estimated activation of LRN layer.

    :param Numlike input_layer: Numlike input
    :param integer local_range: size of local range in local range
                                normalization
    :param integer k: local range normalization k argument
    :param integer alpha: local range normalization alpha argument
    :param integer beta: local range normalization beta argument
    """
    # TODO
    assert_numlike(input_layer)


def pool(layer_input, input_shp, poolsize, stride=(1, 1), mode="max"):
    """Returns estimated activation of max pool layer.

    :param Numlike layer_input: Numlike input
    :param integer pair poolsize: pool of max pool
    :param integer pair stride: stride of max pool
    :param 'max' or 'avg' mode: specifies whether it is max pool or average
                                pool
    """
    assert_numlike(layer_input)
    if mode not in ["max", "avg"]:
        raise ValueError("pool mode should be 'max' or 'avg'")
    is_max = mode == "max"
    # h, w, n_in, n_out - image height, image width, number of input channels,
    #                     number of output channels
    n_in, h, w = input_shp
    n_out = n_in
    # fw, fh, n_out - pool height, pool width
    fh, fw = poolsize
    stride_h, stride_w = stride
    input_type = type(layer_input)
    output_h = (h - fh) / stride_h + 1
    output_w = (w - fw) / stride_w + 1
    output_shp = (n_out, output_h, output_w)
    result = input_type.from_shape(output_shp, neutral=True)
    for at_h in xrange(0, h - fh + 1, stride_h):
        # at_out_h - height of output corresponding to pool at position at_h
        at_out_h = at_h / stride_h
        for at_w in xrange(0, w - fw + 1, stride_w):
            # at_out_w - height of output corresponding to pool at
            # position at_w
            at_out_w = at_w / stride_w
            input_slice = layer_input[:, at_h:(at_h + fh), at_w:(at_w + fw)]
            if is_max:
                pool_res = input_slice.amax(axis=(1, 2), keepdims=False)
            else:
                pool_res = input_slice.sum(axis=(1, 2), keepdims=False) / \
                           float(fh * fw)
            result[:, at_out_h, at_out_w] = pool_res
    return result


def softmax(layer_input):
    """Returns estimated activation of softmax layer."""
    # TODO
    assert_numlike(layer_input)

def relu(layer_input):
    """Returns estimated activation of relu layer."""
    assert_numlike(layer_input)
    try:
        res = layer_input.op_relu()
    except:
        res = (layer_input + layer_input.abs()) * 0.5
    return res
