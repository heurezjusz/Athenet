import numpy

from athenet.algorithm.derest.layers import DerestLayer


class DerestFullyConnectedLayer(DerestLayer):

    def count_derest(self, count_function):
        indicators = numpy.zeros_like(self.layer.W)
        nr_of_batches = self.derivatives.shape.eval()[0]
        for i in range(nr_of_batches):
            act = self.activations.reshape((self.layer.input_shape, 1))
            der = self.derivatives[i].reshape((1, self.layer.output_shape))
            b = (act.dot(der) * self.layer.W).eval()
            indicators = count_function(indicators,b)
        return indicators
