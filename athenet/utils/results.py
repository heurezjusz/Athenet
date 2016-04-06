import numpy
from itertools import product


class Results(object):

    def __init__(self, error_rate, weighted_layers, weights, file=None):
        self.error_rate = error_rate
        self.weighted_layers = numpy.array(weighted_layers)
        self.weights = numpy.array(weights)
        self.file = file
        self.tests = {}

        if file:
            self._load_from_file()

    def add_new_test_result(self, config, zeros, accuracy, save=False):
        self.tests[config] = (zeros, accuracy)
        if save:
            if self.file:
                self._save_to_file()
            else:
                raise "No file to save"

    def _load_from_file(self):
        pass

    def _save_to_file(self):
        pass

    @staticmethod
    def _sum_zeros_in_test(self, (zeros, error_rate), layers=None):
        if layers:
            return sum(z for z, l in zeros[layers])
        return sum(zeros), error_rate

    def _sum_zeros(self, layers=None):
        return numpy.array([self._sum_zeros_in_test(test, layers)
                for test in self.tests])

    def get_zeros_fractions(self, layers=None):
        results = self._get_zeros(layers)
        weights = sum([weights for layer, weights in self.weights[layers]])
        return [(float(zeros) / weights, error_rate) for zeros, error_rate in results]

    def get_zeros_fraction_in_conv_layers(self):
        return self.get_zeros_fractions(self.weighted_layers == "convolutional")

    def get_zeros_fraction_in_fully_connected_layers(self):
        return self.get_zeros_fractions(self.weighted_layers == "fully_connected")

