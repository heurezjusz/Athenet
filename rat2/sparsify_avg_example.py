"""
Functions simple_sparsify_avg_t_example() and simple_sparsify_avg_p_example()
are simple examples for removing weights from graph that are closest to
average value of weights. Are executable.
"""

import tensorflow as tf
import numpy as np
import sparsify_avg as savg

def make_simple_graph():
    """
    Returns simple graph with multiplication.
    """
    g = tf.Graph()
    with g.as_default():
        a = tf.constant(np.random.rand(2, 4))
        v = tf.Variable(np.random.rand(4, 5))
        b = tf.constant(np.random.rand(5, 3))
        o1 = tf.matmul(a, v)
        o2 = tf.matmul(o1, b)
    return g

def simple_sparsify_avg_t_example():
    """
    Example for removing average weights by threshold.
    """
    g = make_simple_graph()
    with g.as_default():
        sess = tf.Session()
        init = tf.initialize_all_variables()
        g_vars = tf.all_variables()
    sess.run(init)
    print "---g_vars before sparsifying---"
    for i in g_vars:
        print i
        print i.eval(sess)
    savg.sparsify_graph_threshold(sess, g, 0.3, savg.mul_var_pred)
    print "---g_vars after sparsifying---"
    for i in g_vars:
        print i
        print i.eval(sess)

def simple_sparsify_avg_p_example():
    """
    Example for removing average weights by percentage.
    """
    g = make_simple_graph()
    with g.as_default():
        sess = tf.Session()
        init = tf.initialize_all_variables()
        g_vars = tf.all_variables()
    sess.run(init)
    print "---g_vars before sparsifying---"
    for i in g_vars:
        print i
        print i.eval(sess)
    savg.sparsify_graph_p(sess, g, 0.3, savg.simple_var_pred)
    print "---g_vars after sparsifying---"
    for i in g_vars:
        print i
        print i.eval(sess)

