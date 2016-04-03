"""Functions that for any neuron calculate its range of derivative of output
with respect to this neuron. Functions should be invoked from the end to the
beginning of the network.

Every estimated impact of tensor on output of network is stored with batches.
Every entity in batches store impact on different output of network.
"""

from athenet.algorithm.numlike import Numlike, assert_numlike

# TODO: All functions below will be implemented.


def d_conv(activation, output):
    # TODO: all
    """Returns estimated impact of convolutional layer on output of network.

    :param Numlike activation: estimated activation of input
    :param Numlike output: estimated impact of output of layer on output
                           of network in shape (batch_size, number of channels,
                           height, width)
    :returns: Estimated impact of input on output of network
    :rtype: Numlike
    """
    assert_numlike(activation)
    assert_numlike(output)


def d_dropout(output, p_dropout):
    """Returns estimated impact of dropout layer on output of network.

    :param Numlike output: estimated impact of output of layer on output
                           of network in shape (batch_size, number of channels,
                           height, width)
    :param float p_dropout: probability of dropping in dropout
    :returns: Estimated impact of input on output of network
    :rtype: Numlike
    """
    assert_numlike(output)
    return output * (1.0 - p_dropout)


def d_fully_connected(output, weights, input_shape):
    # TODO: tests with batches
    """Returns estimated impact of fully connected layer on output of network.

    :param Numlike output: estimated impact of output of layer on output
                           of network in shape (batch_size, number of channels,
                           height, width)
    :param weights: weights of fully connected layer in format (n_in, n_out)
    :type weights: 2D numpy.ndarray or theano.tensor
    :param tuple of integers input_shape: shape of fully connected layer input
    :returns: Estimated impact of input on output of network
    :rtype: Numlike
    """
    assert_numlike(output)
    return output.op_d_fc(weights, input_shape)


def d_norm(output, activation):
    # TODO: all
    """Returns estimated impact of LRN layer on output of network.

    :param Numlike output: estimated impact of output of layer on output
                           of network in shape (batch_size, number of channels,
                           height, width)
    :param Numlike activation: estimated activation of input
    :returns: Estimated impact of input on output of network
    :rtype: Numlike
    """
    assert_numlike(activation)
    assert_numlike(output)


def d_pool(output, activation):
    # TODO: all
    """Returns estimated impact of pool layer on output of network.

    :param Numlike output: estimated impact of output of layer on output
                           of network in shape (batch_size, number of channels,
                           height, width)
    :param Numlike activation: estimated activation of input
    :returns: Estimated impact of input on output of network
    :rtype: Numlike
    """
    assert_numlike(activation)
    assert_numlike(output)
    res = output.op_d_pool(activation)
    return res


def d_softmax(output):
    """Returns estimated impact of softmax layer on output of network.

    .. warning: Current implementation only consider softmax as the last layer.

    :param Numlike output: estimated impact of output of layer on output
                           of network in shape (batch_size, number of channels,
                           height, width)
    :returns: Estimated impact of input on output of network
    :rtype: Numlike
    """
    assert_numlike(output)
    return output


def d_relu(output, activation):
    """Returns estimated impact of relu layer on output of network.

    :param Numlike activation: estimated activation of input
    :param Numlike output: estimated impact of output of layer on output
                           of network in shape (batch_size, number of channels,
                           height, width)
    :returns: Estimated impact of input on output of network
    :rtype: Numlike
    """
    assert_numlike(activation)
    assert_numlike(output)
    res = output.op_d_relu(activation)
    return res
