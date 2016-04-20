class DerestLayer(object):

    def __init__(self, layer):
        self.layer = layer
        self.activations = None
        self.derivatives = None

    def count_activation(self, input):
        raise NotImplementedError

    def count_derivatives(self, output, input_shape):
        #TODO - nice check if activations are done
        raise NotImplementedError

    def count_derest(self, f):
        return []
