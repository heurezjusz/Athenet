import sparsify_avg
from sparsify_avg import sparsify_p
import numpy as np
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
    sparsify_p(c, x)
    print c

for i in range(31):
#   if is_linux():
    os.system("clear")
    f(0.0333333 *  i)
    time.sleep(0.1)
