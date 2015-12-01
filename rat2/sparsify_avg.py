"""
Functions for removing weights from graph that are closest to average value
of weights.
"""

import tensorflow as tf
import numpy as np

def sparsify_pred(m, pred):
    """
    Resets elements of ndarray 'm' for which predicate 'pred' is true.
    args:
        m - numpy.ndarray to be modified
        pred - predicate for modifying 'm'
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

def mul_var_pred():
    """
    Returns predicate that takes variables that are part of some
    multiplication but are not auxiliary gradient descent variables.
    No args.
    """
    g = tf.get_default_graph()
    ops = g.get_operations()
    # operations of multiplications
    op_pred = lambda x: (("MatMul" in x.type or "mul" in x.type)
            and "gradient" not in x.name)
    right_ops = filter(op_pred, ops)
    # any inputs of any multiplication (a * b => a, b are inputs of
    # multiplication operation)
    inputs = [op.inputs for op in right_ops]
    # names of tensors that take part in multiplication and are not just
    # auxiliary for gradient descent
    names = [i.name for a_list in inputs for i in a_list]
    name_pred = lambda x: ("Variable" in x and "gradient" not in x)
    right_names = filter(name_pred, names)
    # checks whether variable has name that is being used as input of a
    # multiplication
    pred = lambda x: x.name in right_names
    return pred

def simple_var_pred():
    """
    Returns predicate that takes variables that are not just auxiliary for
    gradient descent. No args.
    """
    pred = lambda x: ("Variable" in x.name and "gradient" not in x.name)
    return pred

def sparsify_graph_threshold(sess, g, t, pred_factory):
    """
    For any variable in 'g', for session 'sess', resets elements of variable
    that are not more distant from average of this variable than 't'. For
    reseted field f form a variable 'v', (abs(avg(v) - f)) < t.
    args:
        s - session in which we change values
        g - graph in which variables are modified
        t - threshold
        pred_factory - function that returns predicate for variables to reset.
    """
    with g.as_default():
        pred = pred_factory()
        right_vars = filter(pred, tf.all_variables())
        for i in right_vars:
            sess.run(i.initializer)
            new_value = i.eval(sess)
            sparsify_threshold(new_value, t)
            # update 'i' variable
            sess.run(i.assign(new_value))

def sparsify_graph_p(sess, g, p, pred_factory):
    """
    For any variable 'v' in 'g', for session 'sess', resets about (p * size(v))
    fields that are most close to avg(v). Precisely, for
    k = floor(p * size(v)), number of reseted elements is number of elements not
    more distant from average than k'th less distant element.
    args:
        s - session in which we change values
        g - graph in which variables are modified
        p - float, part of matrix to be reseted, must be in range [0, 1]
        pred_factory - function that returns predicate for variables to reset.
    """
    with g.as_default():
        #pred = simple_var_pred()
        pred = pred_factory()
        right_vars = filter(pred, tf.all_variables())
        for i in right_vars:
            sess.run(i.initializer)
            new_value = i.eval(sess)
            sparsify_p(new_value, p)
            sess.run(i.assign(new_value))

