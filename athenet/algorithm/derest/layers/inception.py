from athenet.algorithm.derest.layers import DerestSoftmaxLayer,\
    DerestReluLayer, DerestPoolLayer, DerestNormLayer, DerestLayer, \
    DerestFullyConnectedLayer, DerestConvolutionalLayer, DerestDropoutLayer
from athenet.layers import Softmax, ReLU, PoolingLayer, LRN, \
    ConvolutionalLayer, Dropout, FullyConnectedLayer, InceptionLayer
from athenet.algorithm.derest.utils import add_tuples, change_order


def get_derest_layer(layer, *args):
    """
    Return derest layer on which we can count activations, derivatives
        and derest algorithm

    :param Layer layer: network's original layer
    :return DerestLayer: new better derest layer
    """
    if isinstance(layer, Softmax):
        return DerestSoftmaxLayer(layer, *args)
    if isinstance(layer, ReLU):
        return DerestReluLayer(layer, *args)
    if isinstance(layer, PoolingLayer):
        return DerestPoolLayer(layer, *args)
    if isinstance(layer, LRN):
        return DerestNormLayer(layer, *args)
    if isinstance(layer, ConvolutionalLayer):
        return DerestConvolutionalLayer(layer, *args)
    if isinstance(layer, Dropout):
        return DerestDropoutLayer(layer, *args)
    if isinstance(layer, FullyConnectedLayer):
        return DerestFullyConnectedLayer(layer, *args)
    if isinstance(layer, InceptionLayer):
        return DerestInceptionLayer(layer, *args)
    raise NotImplementedError


class DerestInceptionLayer(DerestLayer):

    def __init__(self, layer, layer_folder, *args):
        super(DerestInceptionLayer, self).__init__(layer, layer_folder, *args)
        self.derest_layer_lists = []
        for (i, layer_list) in zip(xrange(len(self.layer.layer_lists)),
                                   self.layer.layer_lists):
            derest_layer_list = []
            for (j, l) in zip(xrange(len(layer_list)), layer_list):
                folder = self.layer_folder + "/" + str(i) + "/" + str(j)
                derest_layer_list.append(get_derest_layer(l, folder, *args))
            self.derest_layer_lists.append(derest_layer_list)

    def _count_activation(self, input):
        results = None

        for derest_layer_list in self.derest_layer_lists:
            inp = input
            for derest_layer in derest_layer_list:
                inp = derest_layer.count_activation(inp)

            if results is None:
                results = inp
            else:
                results = results.concat(inp)

        return results

    def _count_derivatives(self, output, input_shape):
        output_list = []
        last = 0
        for layer in self.layer.top_layers:
            channels = layer.output_shape[2]
            output_list.append(output[:, last:(last + channels), ::])
            last += channels

        batches = input_shape[0]
        result = None
        for output, derest_list in zip(output_list, self.derest_layer_lists):
            out = output

            for derest_layer in reversed(derest_list):
                out = derest_layer.count_derivatives(out, batches)

            if result is None:
                result = out
            else:
                result += out

        return result

    def count_derest(self, f):
        results = []
        for derest_layer_list in self.derest_layer_lists:
            for derest_layer in derest_layer_list:
                results.extend(derest_layer.count_derest(f))
        return results
