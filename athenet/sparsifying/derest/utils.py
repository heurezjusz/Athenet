"""Auxiliary functions for derest."""

from athenet.sparsifying.utils import Numlike
import theano
import theano.tensor as T
import numpy as np

def assert_numlike(value):
    if isinstance(value, (Numlike, np.ndarray, np.generic)):
        return
    if type(value).__module__ == theano.tensor.__name__:
        return
    print value, type(value)
    raise ValueError("layer_input must be Numlike.")
