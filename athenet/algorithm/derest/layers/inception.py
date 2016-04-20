import theano.tensor as T
from athenet.algorithm.derest.layers.layer import DerestLayer
from athenet.algorithm.derest.utils import get_derest_layer


class DerestInceptionLayer(DerestLayer):

    def __init__(self, layer):
        super(DerestInceptionLayer, layer)
        self.derest_layer_lists = []
        for layer_list in self.layer.layers_lists:
            derest_layer_list = []
            for l in layer_list:
                derest_layer_list.append(get_derest_layer(l))
            self.derest_layer_lists.append(derest_layer_list)

    def count_activation(self, input):
        results = []
        for derest_layer_list in self.derest_layer_lists:
            inp = input
            for derest_layer in derest_layer_list:
                derest_layer.activation = inp
                inp = derest_layer.count_activation(inp)
            results.append(inp)
        return T.concatenate(results)

    def count_derivatives(self, output, input_shape):
        assert NotImplementedError

    def count_derest(self, f):
        raise NotImplementedError
