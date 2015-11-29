import tensorflow as tf
import numpy as np

def sparsify_pred(m, pred):
    """
    Resets elements of 'm' for which predicate 'pred' is true.
    """
    for x in np.nditer(m, op_flags=['readwrite']):
        if pred(x):
            x[...] = 0


def sparsify_threshold(m, t):
    """
    Resets elements of 'm' that are not more distant from average of 'm'
    than t. For reseted field f, (abs(avg(m) - f)) < t
    args:
        m - numpy.ndarray to be modified
        t - float, threshold. Must be greater than 0.
    """
    if t <= 0:
        raise ValueError("t <= 0 but t must be greater than 0")
    avg = np.average(m)
    # pred is true for elements to be deleted
    pred = lambda x: (abs(x - avg) <= t)
    sparsify_pred(m, pred)

def sparsify_p(m, p):
    """
    Resets about 'p' * size('m') elements of 'm' closest to average of 'm'.
    Precisely, for k = floor(p * size(m)), number of reseted elements is number
    of elements not more distant from average than k'th less distant element.
    args:
        m - numpy.ndarray to be modified
        p - float, part of matrix to be reseted, must be in range [0, 1]
    """
    if p < 0:
        raise ValueError("p < 0 but p must be in range [0, 1]")
    if p > 1:
        raise ValueError("p > 1 but p must be in range [0, 1]")
    avg = np.average(m)
    # diff_f returns distant of x from average
    diff_f = np.vectorize(lambda x: abs(x - avg))
    # diffs is array of distances from average
    diffs = diff_f(m).flatten()
    # k is number of elements to be reseted
    k = int(m.size * p)
    # ndarray.partition(-1) sorts array, we don't want it
    if k <= 0:
        return
    # this partition puts (k-1)'th element on (k-1)'th position, counted from 0
    diffs.partition(k - 1)
    threshold = diffs[k - 1]
    sparsify_threshold(m, threshold)

#def sparsify_graph_threshold(g, t):
#    with g.as_default():
#        # trainable variables in g
#        t_vars = tf.trainable_variables()
#        for i in t_vars:

