"""Functions that for any neuron calculate its range of derivative of output
with respect to this neuron. Functions should be invoked from the end to the
beginning of the network.

Every estimated impact of tensor on output of network is stored with batches.
Every entity in batches store impact on different output of network.
"""

from athenet.layers import ConvolutionalLayer, FullyConnectedLayer, \
    InceptionLayer, Dropout, LRN, PoolingLayer, Softmax, ReLU

# TODO: All functions below will be implemented.


def count_derivative(layer_output, activations, input_shape, layer):
    if isinstance(layer, LRN):
        return d_norm(
            layer_output, activations, input_shape,
            layer.local_range, layer.k, layer.alpha,
            layer.beta
        )
    elif isinstance(layer, PoolingLayer):
        return d_pool(
            layer_output, activations, input_shape,
            layer.poolsize, layer.stride, layer.padding,
            layer.mode
        )
    elif isinstance(layer, Softmax):
        return d_softmax(layer_output)
    elif isinstance(layer, ReLU):
        return d_relu(layer_output, activations)
    else:
        raise NotImplementedError

