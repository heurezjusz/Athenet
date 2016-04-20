from athenet.algorithm.derest.layers.layer import DerestLayer


class DerestInceptionLayer(DerestLayer):
    def count_activation(self, input):
        assert NotImplementedError

    def count_derivatives(self, output, input_shape):
        assert NotImplementedError

    def count_derest(self, f):
        raise NotImplementedError
