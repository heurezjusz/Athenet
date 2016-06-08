import numpy

from athenet.utils.misc import save_data_to_pickle, load_data_from_pickle


class Results(object):
    """
    Test results, obtained on one network and using one algorithm
    """

    def __init__(self, error_rate=None, weighted_layers=None,
                 number_of_weights=None, file_=None):
        """
        :param float error_rate: error rate in original network
        :param list of strings weighted_layers:
            network's weighted layers types
        :param list of ints number_of_weights:
            number of weights in weighted layers
        :param string file: file from which load already done tests
        """
        self.error_rate = error_rate
        self.weighted_layers = numpy.array(weighted_layers)
        self.number_of_weights = numpy.array(number_of_weights)
        self.tests = {}

        self.file = file_
        if file_:
            self.load_from_file()

    def add_new_test_result(self, config, result, save=False):
        """
        Adds new test result

        :param float or iterable config: parameters used to test network
        :param tuple(list, float) result:
            list of number of zeros in every layers
            and error_rate in tested network
        :param bool save: whenever save it  to file
        """
        self.tests[config] = result
        if save:
            self.save_to_file()

    def load_from_file(self, file_=None):
        """
        Adds test results from file. If file is not given,
         will use default file set in init.

        :param string file_: file to load from
        """
        if file_ is None:
            file_ = self.file
        if file_ is None:
            raise "No file to load from"

        try:
            loaded_data = load_data_from_pickle(file_)

            if self.error_rate is None:
                self.error_rate = loaded_data.error_rate
                self.weighted_layers = loaded_data.weighted_layers
                self.number_of_weights = loaded_data.number_of_weights

            self.tests = dict(self.tests, **loaded_data.tests)
        except IOError:
            self.save_to_file(file_)

    def save_to_file(self, file=None):
        """
        Saves everything to file. If file is not given,
         will use default file set in init

        :param string file: file to save to
        :return:
        """
        if file is None:
            file = self.file
        if file is None:
            raise "No file to save to"
        save_data_to_pickle(self, file)

    def get_new_test_configs(self, configs):
        """
        Returns test cases which are not yet checked

        :param list or tuple configs: parameters for tests
        :return list: parameters for tests not yet checked
        """
        return [config for config in configs if config not in self.tests]

    def _get_number_of_weights(self, layers=None):
        if layers is not None:
            return self.number_of_weights[layers]
        return self.number_of_weights

    @staticmethod
    def _sum_zeros_in_test((zeros, error_rate), layers=None):
        if layers is not None:
            return sum(z for z in zeros[layers]), error_rate
        return sum(zeros), error_rate

    def _sum_zeros(self, layers=None):
        return {config: self._sum_zeros_in_test(test, layers)
                for config, test in self.tests.iteritems()}

    def get_zeros_fraction(self, layers=None):
        """
        Counts fraction of zeros on layers for every test

        :param layers: which layers to consider
        :type layers: list of bools or tuple of bools
        :return list of tuples(float, float):
            fractions of zeros and error rate for every test
        """
        results = self._sum_zeros(layers)
        number_of_weights = sum(self._get_number_of_weights(layers))
        return {config: (float(zeros) / number_of_weights, error_rate)
                for config, (zeros, error_rate) in results.iteritems()}

    def get_zeros_fraction_in_conv_layers(self):
        """
        Counts fraction of zeros on convolutional layers for every test

        :return list of tuples(float, float):
            fractions of zeros and error rate for every test
        """
        conv_layers = self.weighted_layers == "ConvolutionalLayer"
        return self.get_zeros_fraction(conv_layers)

    def get_zeros_fraction_in_fully_connected_layers(self):
        """
        Counts fraction of zeros on convolutional layers for every test

        :return list of tuples(float, float):
            fractions of zeros and error rate for every test
        """
        fully_connected_layers = self.weighted_layers == "FullyConnectedLayer"
        return self.get_zeros_fraction(fully_connected_layers)
