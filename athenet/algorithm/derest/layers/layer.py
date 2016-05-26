from athenet.algorithm.derest.utils import change_order, add_tuples,\
    make_iterable

class DerestLayer(object):

    def __init__(self, layer):
        self.layer = layer
        self.activations = None
        self.derivatives = None

    @staticmethod
    def _normalize(data):
        a = data.abs().amax().upper
        return data / a

    def _count_activation(self, layer_input, normalize=False):
        raise NotImplementedError

    def count_activation(self, layer_input, normalize=False):
        """
        Returns estimated activations

        :param Numlike layer_input:
        :param boolean normalize: whenever normalize number between layers
        :return Numlike:
        """
        if normalize:
            layer_input = self._normalize(layer_input)
        input_shape = change_order(make_iterable(self.layer.input_shape))
        layer_input = layer_input.reshape(input_shape)
        self.activations = layer_input
        return self._count_activation(layer_input, normalize)

    def _count_derivatives(self, layer_output, input_shape, normalize=False):
        raise NotImplementedError

    def count_derivatives(self, layer_output, batches, normalize=False):
        """
        Returns estimated impact of input of layer on output of network

        :param Numlike layer_output:
        :param tuple input_shape:
        :param boolean normalize: whenever normalize number between layers
        :return Numlike:
        """
        if normalize:
            layer_output = self._normalize(layer_output)
        input_shape = add_tuples(batches,
                                 change_order(self.layer.input_shape))
        output_shape = add_tuples(batches,
                                  change_order(self.layer.output_shape))
        layer_output = layer_output.reshape(output_shape)
        self.derivatives = layer_output
        return self._count_derivatives(layer_output, input_shape, normalize)

    def count_derest(self, count_function):
        """
        Returns indicators of each weight importance

        :param function count_function: function to count indicators,
            takes Numlike and returns float
        :return list of numpy arrays:
        """
        return []
