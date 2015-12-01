import tensorflow as tf
import numpy as np
import sparsify_avg as savg
import time
import os
from sys import platform as _platform

m = np.random.rand(5, 4)

def is_linux():
    if _platform == "linux" or _platform == "linux2":
        return True
    else:
        return False

def f(x):
    c = m.copy()
    savg.sparsify_p(c, x)
    print c

def sparsify_p_example():
    for i in range(31):
    #   if is_linux():
        os.system("clear")
        f(0.0333333 *  i)
        time.sleep(0.1)

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

def simple_sparsify_avg_test():
    g = make_simple_graph()
    with g.as_default():
        sess = tf.Session()
        init = tf.initialize_all_variables()
        g_vars = tf.all_variables()
    sess.run(init)
    print "---g_vars before sparsifying---"
    for i in g_vars:
        print i.eval(sess)
    savg.sparsify_graph_threshold(sess, g, 0.3)
    print "---g_vars after sparsifying---"
    for i in g_vars:
        print i
        print i.eval(sess)
