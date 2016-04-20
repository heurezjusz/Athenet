import theano.tensor as T
from athenet.algorithm.derest.layers.layer import DerestLayer
from athenet.algorithm.derest.utils import get_derest_layer


class DerestInceptionLayer(DerestLayer):

    def __init__(self, layer):
        super(DerestInceptionLayer, self).__init__(layer)
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
        return T.concatenate(results, axis=1)

    def count_derivatives(self, output, input_shape):
        output_list = []
        last = 0
        for layer in self.layer.top_layers:
            width = layer.output_shape[1]
            output_list.append(output[:, last : (last + width), ::])
            last += width

        result = None
        for output, derest_list in zip(output, self.derest_layer_lists):
            out = output
            for derest_layer in reversed(derest_list):
                out = derest_list.count_derivatives(out)
                derest_layer.derivatives = out
            if result is None:
                result = out
            else:
                result += out

        return result

    def count_derest(self, f):
        results = []
        for derest_layer_list in self.derest_layer_lists:
            for derest_layer in derest_layer_list:
                results += derest_layer.count_derest(f)
        return results
