import numpy


def get_random_indicators(layers):
    return [numpy.random.rand(layer.W.size).reshape(layer.W.shape)
            for layer in layers]
