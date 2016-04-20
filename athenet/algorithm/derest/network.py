from athenet.algorithm.derest.utils import _change_order, add_tuples
from athenet.algorithm.derest.layer import derest_layer


class DerestNetwork():

    def __init__(self, network):
        self.network = network
        self.layers = [derest_layer(layer)
                       for layer in network.layers]

    def _get_layer_input_shape(self, i):
        if i > 0:
            return self.layers[i - 1].layer.output_shape
        return self.layers[i].layer.input_shape

    def count_activations(self, inp):
        for layer in self.layers:
            layer.activations = inp
            inp = layer.count_activation(inp)
        return inp

    def count_derivatives(self, outp):
        batches = outp.shape.eval()[0]
        for i in range(len(self.layers) - 1, -1, -1):
            input_shape = add_tuples(
                batches,
                _change_order(self._get_layer_input_shape(i))
            )
            outp = self.layers[i].count_derivatives(
                outp,
                input_shape
            )
        return outp

    def count_derest(self, f):
        result = []
        for layer in self.layers:
            indices = layer.count_derest(f)
            if indices is not None:
                result.append(indices)
        return result
