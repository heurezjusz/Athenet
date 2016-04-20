import numpy


class NetworkMock(object):
    def __init__(self, weighted_layers=None):

        if weighted_layers is None:
            weighted_layers = np.array()

        self.weighted_layers = weighted_layers
        self.layers = weighted_layers


class LayerMock(object):
    def __init__(self, weights=None, biases=None):
        self.W = weights if weights is not None else numpy.array()
        self.B = biases if biases is not None else numpy.array()
