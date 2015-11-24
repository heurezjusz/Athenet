"""
Program for training and saving net to a file.
"""

import cPickle
import LeNet.convolutional_mlp as conv

weights, biases = cPickle.load(open('conv.pkl'))
conv.test_lenet5(weights, biases)