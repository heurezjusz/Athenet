from athenet.algorithm.derest.derest import derest
from athenet.network import Network
from athenet.layers import ConvolutionalLayer, FullyConnectedLayer,\
    Softmax, ReLU, MaxPool, InceptionLayer, LRN


def aggregating_example_function(values):
    a = values.sum()
    return a.lower + a.upper

n = Network([
    ConvolutionalLayer(image_shape=(28, 28, 1), filter_shape=(4, 4, 2)),
    ReLU(),
    LRN(),
    MaxPool(poolsize=(2, 2)),
#    InceptionLayer(n_filters=[2, 2, 2, 2, 2, 2]),
    FullyConnectedLayer(n_out=10),
    ReLU(),
    FullyConnectedLayer(n_out=3),
    Softmax(),
])

derest(n, 0.6, count_function=aggregating_example_function, max_batch_size=None,
       normalize_activations=True, normalize_derivatives=True)