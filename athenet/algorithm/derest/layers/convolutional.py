from itertools import product

import numpy

from athenet.algorithm.derest.layers.layer import DerestLayer


class DerestConvolutionalLayer(DerestLayer):

    def _get_activation_for_weight(self, i1, i2, i3):
        #no padding or strides yet considered
        n1, n2, _ = self.layer.input_shape
        m1, m2, _ = self.layer.filter_shape
        return self.activations[i1, i2:(n1-m2+i2+1), i3:(n2-m2+i3+1)]

    def count_derest(self, f):
        indicators = numpy.zeros_like(self.layer.W)

        i0, i1, i2, i3 = self.layer.W.shape
        for batch_nr in range(self.derivatives.shape.eval()[0]): #for every batch
            der = self.derivatives[batch_nr]
            for j1, j2, j3, j4 in product(range(i0), range(i1), range(i2), range(i3)):
                y = self._get_activation_for_weight(j2, j3, j4)
                x = (der[j1] * y * self.layer.W[j1, j2, j3, j4]).eval()
                indicators[j1, j2, j3, j4] = f(indicators[j1, j2, j3, j4], x, True)

        return indicators
