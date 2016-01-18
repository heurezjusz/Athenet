import numpy as np

class Filter(object):
    """Filters used to modify convolutional layer's weights
    """
    def apply(self, table):
        """:table: numpy 2D table represents part of convolutional layer's
           weights (one filter). Function modifies elements of the table.
        """
        raise NotImplementedError

class WeightedFilter(Filter):
    def __init__(self, W):
        self.W = W

    def adjust(self, i, j, shape):
        if i >= 0 and i < shape[0] and j >= 0 and j < shape[1]:
            return i, j

        if i < 0:
            if j < 0:
                return adjust(i + 1, j + 1, shape)
            return adjust(i + 1, j, shape)
        if j < 0:
            return adjust(i, j + 1, shape)

        if i >= shape[0]:
            if j >= shape[1]:
                return adjust(i - 1, j - 1, shape)
            return adjust(i - 1, j, shape)
        return adjust(i, j - 1, shape)


    def apply(self, table):
        result = np.zeros(talbe.shape)
        sum = np.sum(self.W)
        for i in xrange(table.shape[0]):
            for j in xrange(table.shape[1]):
                for fi in xrange(W.shape[0]):
                    for fj in xrange(W.shape[1]):
                        tmpi, tmpj = self.adjust(i - W.shape[0] / 2 + fi,
                                                 j - W.shape[1] / 2 + fj,
                                                 table.shape)
                        result[i][j] += self.W[fi][fj] * table[tmpi][tmpj]
        for i in xrange(table.shape[0]):
            for j in xrange(table.shape[1]):
                table[i][j] = result[i][j]
                if sum != 0:
                    table[i][j] /= sum


avg_filter = WeightedFilter(np.asarray(
    [[ 1., 1., 1.],
     [ 1., 1., 1.],
     [ 1., 1., 1.]]
))


hp3_filter = WeightedFilter(np.asarray(
    [[  0., -1.,  0.],
     [ -1., 20., -1.],
     [  0., -1.,  0.]]
))


vertical_edge_filter = WeightedFilter(np.asarray(
    [[  0., 0., 0. ],
     [ -1., 1., 0. ],
     [  0., 0., 0. ]]
))


horizontal_edge_filter = WeightedFilter(np.asarray(
    [[ 0., -1., 0. ],
     [ 0.,  1., 0. ],
     [ 0.,  0., 0. ]]
))