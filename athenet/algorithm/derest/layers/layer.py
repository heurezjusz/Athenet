from athenet.algorithm.derest.utils import change_order, add_tuples,\
    make_iterable


class DerestLayer(object):

    def __init__(self, layer, normalize_activation=lambda x: x,
                 normalize_derivatives=lambda x: x):
        self.layer = layer
        self.activations = None
        self.derivatives = None
        self._normalize_activation = normalize_activation
        self._normalize_derivatives = normalize_derivatives

    def _count_activation(self, layer_input):
        raise NotImplementedError

    def count_activation(self, layer_input):
        """
        Returns estimated activations

        :param Numlike layer_input:
        :return Numlike: activations
        """
        layer_input = self._normalize_activation(layer_input)
        input_shape = change_order(make_iterable(self.layer.input_shape))
        layer_input = layer_input.reshape(input_shape)
        self.activations = layer_input
        return self._count_activation(layer_input)

    def _count_derivatives(self, layer_output, input_shape):
        raise NotImplementedError

    def count_derivatives(self, layer_output, batches):
        """
        Returns estimated impact of input of layer on output of network

        :param Numlike layer_output:
        :param tuple input_shape: shape of input
        :param int batches: number of batches
        :return Numlike: derivatives
        """
        layer_output = self._normalize_derivatives(layer_output)
        input_shape = add_tuples(batches,
                                 change_order(self.layer.input_shape))
        output_shape = add_tuples(batches,
                                  change_order(self.layer.output_shape))
        layer_output = layer_output.reshape(output_shape)

        if self.derivatives is not None:
            self.derivatives = self.derivatives.concat(layer_output)
        else:
            self.derivatives = layer_output

        return self._count_derivatives(layer_output, input_shape)

    def count_derest(self, count_function):
        """
        Returns indicators of each weight importance

        :param function count_function: function to count indicators,
            takes Numlike and returns float
        :return list of numpy arrays:
        """
        return []
