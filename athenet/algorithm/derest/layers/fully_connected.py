import numpy
from itertools import product

from athenet.algorithm.derest.layers import DerestLayer
from athenet.algorithm.numlike import assert_numlike


class DerestFullyConnectedLayer(DerestLayer):

    def _count_activation(self, layer_input):
        """
        Return estimated activations

        :param Numlike layer_input: input for layer
        :return Numlike:
        """
        return a_fully_connected(layer_input, self.layer.W, self.layer.b)

    def _count_derivatives(self, output, input_shape):
        """
        Returns estimated impact of input of layer on output of
        network.

        :param Numlike layer_output: impact of input of next layer
            on output of network
        :param tuple input_shape:
        :return Numlike:
        """
        return d_fully_connected(output, self.layer.W, input_shape)

    def count_derest(self, count_function):
        """
        Returns indicators of each weight importance

        :param function count_function: function to count indicators,
            takes Numlike and returns float
        :return list of numpy arrays:
        """
        input_shape = self.layer.input_shape
        output_shape = self.layer.output_shape

        derivatives = self.load_derivatives()
        batches = derivatives.shape[0]
        activations = self.load_activations().reshape((input_shape, 1)).\
            broadcast((batches, input_shape, 1))
        derivatives = derivatives.reshape((batches, 1, output_shape))

        ind = ((activations * derivatives).sum((0,)) * self.layer.W)
        return [count_function(ind)]


def a_fully_connected(layer_input, weights, biases):
    """Returns estimated activation of fully connected layer.

    :param Numlike layer_input: input Numlike
    :param weights: weights of fully connected layer in format
    (n_in, n_out)
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


def d_fully_connected(output, weights, input_shape):
    """Returns estimated impact of input of fully connected layer on output of
    network.

    :param Numlike output: estimated impact of output of layer on output
                           of network in shape (batch_size, number of channels,
                           height, width)
    :param weights: weights of fully connected layer in format (n_in, n_out)
    :type weights: 2D numpy.ndarray or theano.tensor
    :param input_shape: shape of fully connected layer input in any format with
                        number of batches at the beginning.
    :type input_shape: tuple of integers
    :returns: Estimated impact of input on output of network
    :rtype: Numlike
    """
    assert_numlike(output)
    try:
        res = output.dot(weights.T)
    except NotImplementedError:
        res = (output * weights).sum(1)
    return res.reshape(input_shape)
