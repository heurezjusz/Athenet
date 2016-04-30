from athenet.algorithm.derest.layers import DerestSoftmaxLayer,\
    DerestReluLayer, DerestPoolLayer, DerestNormLayer, DerestLayer, \
    DerestFullyConnectedLayer, DerestConvolutionalLayer, DerestDropoutLayer
from athenet.layers import Softmax, ReLU, PoolingLayer, LRN, \
    ConvolutionalLayer, Dropout, FullyConnectedLayer, InceptionLayer
from athenet.algorithm.derest.utils import derest_normalize


def get_derest_layer(layer, normalize=False):
    """
    Return derest layer on which we can count activations, derivatives
        and derest algorithm

    :param Layer layer: network's original layer
    :return DerestLayer: new better derest layer
    """
    if isinstance(layer, Softmax):
        return DerestSoftmaxLayer(layer)
    if isinstance(layer, ReLU):
        return DerestReluLayer(layer)
    if isinstance(layer, PoolingLayer):
        return DerestPoolLayer(layer)
    if isinstance(layer, LRN):
        return DerestNormLayer(layer)
    if isinstance(layer, ConvolutionalLayer):
        return DerestConvolutionalLayer(layer)
    if isinstance(layer, Dropout):
        return DerestDropoutLayer(layer)
    if isinstance(layer, FullyConnectedLayer):
        return DerestFullyConnectedLayer(layer)
    if isinstance(layer, InceptionLayer):
        return DerestInceptionLayer(layer)
    raise NotImplementedError


class DerestInceptionLayer(DerestLayer):

    def __init__(self, layer):
        super(DerestInceptionLayer, self).__init__(layer)
        self.derest_layer_lists = []
        for layer_list in self.layer.layer_lists:
            derest_layer_list = []
            for l in layer_list:
                derest_layer_list.append(get_derest_layer(l))
            self.derest_layer_lists.append(derest_layer_list)

    def count_activation(self, input):
        results = None

        for derest_layer_list in self.derest_layer_lists:
            inp = input
            for derest_layer in derest_layer_list:
                #do not work for a moment
                #if self.normalize:
                #    inp = derest_normalize(inp)
                derest_layer.activations = inp
                inp = derest_layer.count_activation(inp)

            if results is None:
                results = inp
            else:
                results = results.concat(inp)

        return [results]

    def count_derivatives(self, output, input_shape):
        output_list = []
        last = 0
        for layer in self.layer.top_layers:
            channels = layer.output_shape[2]
            output_list.append(output[:, last : (last + channels), ::])
            last += channels

        result = None
        result = []
        i = 0
        for output, derest_list in zip(output_list, self.derest_layer_lists):
            if i == 1:
                break
            out = output
            for derest_layer in reversed(derest_list):
                print derest_layer
                derest_layer.derivatives = out
                out = derest_layer.count_derivatives(out, input_shape)
                return out

            result.append(out)
            i += 1
            #if result is None:
            #    result = out
            #else:
            #    result += out

        return result[0]

    def count_derest(self, f):
        results = []
        for derest_layer_list in self.derest_layer_lists:
            for derest_layer in derest_layer_list:
                results += derest_layer.count_derest(f)
        return results
