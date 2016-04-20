from athenet.algorithm.derest.layers.layer import DerestLayer


class DerestInceptionLayer(DerestLayer):
    def count_activation(self, input):
        assert NotImplementedError

    def count_derivatives(self, output, input_shape):
        assert NotImplementedError

    def count_derest(self, f):
        raise NotImplementedError


def a_inception(layer_input, layer):
    """Returns estimated activation of inception layer.

    :param Numlike layer_input: input
    :rtype: Numlike
    :param InceptionLayer layer: layer of which activation will be retuned
    """
    assert_numlike(layer_input)
    assert(isinstance(layer, InceptionLayer))

    out = []
    for layer_list in layer.layer_lists:
        inp = layer_input
        for l in layer_list:
            inp = count_activation(inp, l)
        out.append(inp)

    return T.concatenate(out, axis=1)
