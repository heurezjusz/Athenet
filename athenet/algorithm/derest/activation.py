"""Functions that for any neuron and given range of inputs of layers calculate
neuron's range of activation. Functions are being invoked from the beginning
to the end of the network.
"""

from athenet.algorithm.numlike import Numlike, assert_numlike


def conv(layer_input, image_shape, weights, filter_shape, biases,
         stride=(1, 1), padding=(0, 0), n_groups=1):
    """Returns estimated activation of convolutional layer.

    :param layer_input: input Numlike in input_shp format
    :param image_shape: shape of input in the format
                (number of input channels, image height, image width)
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
        padded_input = input_type.from_shape(padded_input_shape)
        padded_input[0:n_in, pad_h:(pad_h + h), pad_w:(pad_w + w)] = \
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
                # at_out_h - height of output corresponding to filter at
                # position at_h
                at_out_h = at_h / stride_h
                for at_w in xrange(0, w - fw + 1, stride_w):
                    # at_out_w - height of output corresponding to filter at
                    # position at_w
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
    :param weights: weights of fully connected layer in format (n_in, n_out)
    :param biases: biases of fully connected layer of size n_out
    :type weights: 2D numpy.ndarray or theano.tensor
    :type biases: 1D numpy.ndarray or theano.tensor
    :rtype: Numlike
    """
    assert_numlike(layer_input)
    flat_input = layer_input.flatten()
    try:
        return flat_input.dot(weights) + biases
    except NotImplementedError:
        return (flat_input * weights.T).sum(1) + biases


def norm(layer_input, input_shape, local_range=5, k=1, alpha=0.00002,
         beta=0.75):
    """Returns estimated activation of LRN layer.

    :param Numlike layer_input: Numlike input
    :param input_shape: shape of Interval in format (n_channels, height, width)
    :param integer local_range: size of local range in local range
                                normalization
    :param integer k: local range normalization k argument
    :param integer alpha: local range normalization alpha argument
    :param integer beta: local range normalization beta argument
    :type input_shape: tuple of 3 integers
    :rtype: Numlike
    """
    assert_numlike(layer_input)
    try:
        return layer_input.op_norm(input_shape, local_range, k, alpha, beta)
    except NotImplementedError:
        half = local_range / 2
        sq = layer_input.square()
        n_channels, h, w = input_shape
        extra_channels = layer_input.from_shape((n_channels + 2 * half, h, w),
                                                neutral=True)
        extra_channels[half:half + n_channels, :, :] = sq
        local_sums = layer_input.from_shape(input_shape, neutral=True)

        for i in xrange(local_range):
            local_sums += extra_channels[i:i + n_channels, :, :]

        return layer_input / (local_sums * alpha + k).power(beta)


def pool(layer_input, input_shp, poolsize, stride=(1, 1), mode="max"):
    """Returns estimated activation of pool layer.

    :param Numlike layer_input: Numlike input in input_shp format
    :param tuple of 3 integers input_shp: input shape in format (n_channels,
                                          height, width)
    :param pair of integers poolsize: pool size in format (height, width)
    :param pair of integers stride: stride of max pool
    :param 'max' or 'avg' mode: specifies whether it is max pool or average
                                pool
    :rtype: Numlike
    """
    assert_numlike(layer_input)
    if mode not in ["max", "avg"]:
        raise ValueError("pool mode should be 'max' or 'avg'")
    is_max = mode == "max"
    # n_in, h, w - number of input channels, image height, image width
    n_in, h, w = input_shp
    n_out = n_in
    # fh, fw - pool height, pool width
    fh, fw = poolsize
    stride_h, stride_w = stride
    output_h = (h - fh) / stride_h + 1
    output_w = (w - fw) / stride_w + 1
    output_shp = (n_out, output_h, output_w)
    result = layer_input.from_shape(output_shp, neutral=True)
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


def softmax(layer_input, input_shp):
    """Returns estimated activation of softmax layer.
    :param Numlike layer_input: input
    :param integer input_shp: shape of 1D input
    :rtype: Numlike
    """
    assert_numlike(layer_input)
    try:
        res = layer_input.op_softmax(input_shp)
    except NotImplementedError:
        exponents = layer_input.exp()
        res = exponents / exponents.sum()
    return res


def relu(layer_input):
    """Returns estimated activation of relu layer.

    :param Numlike layer_input: input
    :rtype: Numlike
    """
    assert_numlike(layer_input)
    try:
        res = layer_input.op_relu()
    except NotImplementedError:
        res = (layer_input + layer_input.abs()) * 0.5
    return res
