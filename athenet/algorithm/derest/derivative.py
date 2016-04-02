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


def d_fully_connected(activation, weights, output):
    # TODO: tests with batches
    """Returns estimated impact of fully connected layer on output of network.

    :param Numlike activation: estimated activation of input
    :param weights: weights of fully connected layer in format (n_in, n_out)
    :param Numlike output: estimated impact of output of layer on output
                           of network in shape (batch_size, number of channels,
                           height, width)
    :type weights: 2D numpy.ndarray or theano.tensor
    :returns: Estimated impact of input on output of network
    :rtype: Numlike
    """
    assert_numlike(activation)
    assert_numlike(output)
    try:
        return output.dot(weights.T)
    except NotImplementedError:
        return (output * weights).sum(1)


def d_norm(activation, output):
    # TODO: all
    """Returns estimated impact of LRN layer on output of network.

    :param Numlike activation: estimated activation of input
    :param Numlike output: estimated impact of output of layer on output
                           of network in shape (batch_size, number of channels,
                           height, width)
    :returns: Estimated impact of input on output of network
    :rtype: Numlike
    """
    assert_numlike(activation)
    assert_numlike(output)


def d_pool(activation, output):
    # TODO: all
    """Returns estimated impact of max pool layer on output of network.

    :param Numlike activation: estimated activation of input
    :param Numlike output: estimated impact of output of layer on output
                           of network in shape (batch_size, number of channels,
                           height, width)
    :returns: Estimated impact of input on output of network
    :rtype: Numlike
    """
    assert_numlike(activation)
    assert_numlike(output)


def d_softmax(activation, output):
    # TODO: all
    """Returns estimated impact of softmax layer on output of network.

    .. warning: Current implementation only consider softmax as the last layer.

    :param Numlike activation: estimated activation of input
    :param Numlike output: estimated impact of output of layer on output
                           of network in shape (batch_size, number of channels,
                           height, width)
    :returns: Estimated impact of input on output of network
    :rtype: Numlike
    """
    assert_numlike(activation)
    assert_numlike(output)
    return output


def d_relu(activation, output):
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
