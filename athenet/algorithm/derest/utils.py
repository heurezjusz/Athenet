from athenet.algorithm.derest.layers import DerestSoftmaxLayer,\
    DerestReluLayer, DerestPoolLayer, DerestNormLayer,\
    DerestConvolutionalLayer, DerestDropoutLayer, DerestFullyConnectedLayer,\
    DerestInceptionLayer
from athenet.layers import Softmax, ReLU, PoolingLayer, LRN, \
    ConvolutionalLayer, Dropout, FullyConnectedLayer, InceptionLayer


def _change_order(a):
    """
    So the last will be first
    """
    try:
        h, w, n = a
        return (n, h, w)
    except:
        return a


def add_tuples(a, b):
    if not isinstance(a, tuple):
        a = (a, )
    if not isinstance(b, tuple):
        b = (b, )
    return a + b


def get_derest_layer(layer):
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

