from athenet.algorithm.numlike import NpInterval
import numpy as np
from unittest import TestCase, main
from random import randrange

class TestShape(TestCase):
    def _run_test(self, shape):
        i = NpInterval(np.zeros(shape), np.ones(shape))
        self.assertEquals(shape, i.shape)

    def _random_shape(self):
        result = None
        limit = 10 ** 4
        size = 1
        for i in xrange(randrange(1,7)):
            l = randrange(1,10)
            if result is None:
                result = (l,)
            else:
                result += (l,)
            size *= l
            if size >= limit:
                return result
        return result

    def test_shape(self):
        for i in xrange(100):
            self._run_test(self._random_shape())


class Just(TestCase):
    def test(self):
        shape = (2, 5, 3, 3)
        act = NpInterval(np.ones(shape), np.ones(shape) * 2)
        norm = act.op_d_norm(act, shape, 5, 1, 1, 0.5)
        print act
        print norm

if __name__ == '__main__':
    main(verbosity=2)