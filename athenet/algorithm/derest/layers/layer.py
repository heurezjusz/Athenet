class DerestLayer(object):

    def __init__(self, layer):
        self.layer = layer
        self.activations = None
        self.derivatives = None
        self.normalize = False

    def count_activation(self, layer_input):
        """
        Returns estimated activations

        :param Numlike layer_input:
        :return Numlike:
        """
        raise NotImplementedError

    def count_derivatives(self, layer_output, input_shape):
        """
        Returns estimated impact of input of layer on output of network

        :param Numlike layer_output:
        :param tuple input_shape:
        :return Numlike:
        """
        raise NotImplementedError

    def count_derest(self, count_function):
        """
        Returns indicators of each weight importance

        :param function count_function: function to count indicators,
            takes Numlike and returns float
        :return list of numpy arrays:
        """
        return []
