import numpy

from athenet.utils.misc import save_data_to_pickle, load_data_from_pickle


class Results(object):

    def __init__(self, error_rate, weighted_layers, weights, file=None):
        self.error_rate = error_rate
        self.weighted_layers = numpy.array(weighted_layers)
        self.weights = numpy.array(weights)
        self.tests = {}

        self.file = file
        if file:
            self.load_from_file()

    def add_new_test_result(self, config, result, save=False):
        self.tests[config] = result
        if save:
            if self.file:
                self.save_to_file()
            else:
                raise "No file to save"

    def load_from_file(self, file=None):
        if file is None:
            file = self.file
        loaded_data = load_data_from_pickle(file)

        assert loaded_data.error_rate == self.error_rate
        assert numpy.array_equal(loaded_data.weighted_layers,
                                     self.weighted_layers)
        assert numpy.array_equal(loaded_data.weights, self.weights)
        self.tests = dict(self.tests, **loaded_data.tests)

    def save_to_file(self, file=None):
        if file is None:
            file = self.file
        save_data_to_pickle(self, file)

    def get_new_test_configs(self, configs):
        return [config for config in configs if config not in self.tests]

    def _get_weights(self, layers=None):
        if layers is not None:
            return self.weights[layers]
        return self.weights

    @staticmethod
    def _sum_zeros_in_test((zeros, error_rate), layers=None):
        if layers is not None:
            return sum(z for z in zeros[layers]), error_rate
        return sum(zeros), error_rate

    def _sum_zeros(self, layers=None):
        return numpy.array([self._sum_zeros_in_test(test, layers)
                for test in self.tests.itervalues()])

    def get_zeros_fraction(self, layers=None):
        results = self._sum_zeros(layers)
        weights = sum(self._get_weights(layers))
        return [(float(zeros) / weights, error_rate)
                for zeros, error_rate in results]

    def get_zeros_fraction_in_conv_layers(self):
        conv_layers = self.weighted_layers == "ConvolutionalLayer"
        return self.get_zeros_fraction(conv_layers)

    def get_zeros_fraction_in_fully_connected_layers(self):
        fully_connected_layers = self.weighted_layers == "FullyConnectedLayer"
        return self.get_zeros_fraction(fully_connected_layers)

