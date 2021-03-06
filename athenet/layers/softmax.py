"""Softmax layer."""

import theano.tensor as T
from theano.tensor.nnet import softmax

from athenet.layers import Layer


class Softmax(Layer):
    """Softmax layer."""
    def __init__(self, input_layer_name=None, name='softmax'):
        super(Softmax, self).__init__(input_layer_name, name)

    def _get_output(self, layer_input):
        """Return layer's output.

        :param layer_input: Input in the format (n_batches, n_neurons).
        :return: Layer output.
        """
        return softmax(layer_input)

    def set_cost(self, answer):
        """Set layer's cost function.

        :param answer: Vector of desired answers for minibatch.
        """
        self.cost = T.mean(-T.log(self.train_output)
                           [T.arange(answer.shape[0]), answer])
