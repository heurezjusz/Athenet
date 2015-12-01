import sparsify_avg
from sparsify_avg import sparsify_p
import numpy as np
import time
import os
from sys import platform as _platform
import tensorflow as tf

m = np.random.rand(5, 4)

def is_linux():
    if _platform == "linux" or _platform == "linux2":
        return True
    else:
        return False

def f(x):
    c = m.copy()
    sparsify_p(c, x)
    print c

def sparsify_p_example():
    for i in range(31):
    #   if is_linux():
        os.system("clear")
        f(0.0333333 *  i)
        time.sleep(0.1)

def make_simple_graph():
    """
    Returns simple graph.
    """
    g = tf.Graph()
    with g.as_default():
        a = tf.constant(4)
        b = tf.constant(5)
        v = tf.Variable(6)
        o1 = a + v
        o2 = b + v
    return g
