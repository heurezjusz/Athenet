"""
Program for training and saving net to a file.
"""

import cPickle
import LeNet.convolutional_mlp as conv

layers = conv.evaluate_lenet5(n_epochs=1)
f = open('net.pkl', 'w')
weights = [layers[i].W.get_value() for i in xrange(4)]
biases = [layers[i].b.get_value() for i in xrange(4)]
data = [weights, biases]
cPickle.dump(data, f)
