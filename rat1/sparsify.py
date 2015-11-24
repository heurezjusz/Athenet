"""
Program for zeroing 'p' percent smallest neural network's weights.
"""

import cPickle
import LeNet.convolutional_mlp as conv
import numpy as np

p = 0.07

def sparsify_layer(weights, n_layer):
    W = [(weights[n_layer][i], i) for i in np.ndindex(weights[n_layer].shape)]
    W = sorted(W, key=lambda weight_pair: weight_pair[0], reverse=True)
    for i in xrange(int(len(W) * p)):
        weights[n_layer][W[i][1]] = 0
    return weights

print 'p = %.1f%%' % (p * 100)
weights, biases = cPickle.load(open('net_200epochs.pkl'))
print 'Original network:'
conv.test_lenet5(weights, biases)
for i in xrange(4):
    print 'With layer %d sparsified:' % i
    weights2 = sparsify_layer(weights, i)
    conv.test_lenet5(weights2, biases)
