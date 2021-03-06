import pickle
import os

from athenet.algorithm.derest.utils import change_order, add_tuples,\
    make_iterable


class DerestLayer(object):
    need_activation = False
    need_derivatives = False

    def __init__(self, layer, layer_folder, normalize_activation=lambda x: x,
                 normalize_derivatives=lambda x: x):
        self.layer = layer

        self.layer_folder = layer_folder + "_" + layer.name
        if not os.path.exists(self.layer_folder):
            os.makedirs(self.layer_folder)

        self.normalize_activation = normalize_activation
        self._normalize_derivatives = normalize_derivatives

    def normalize_derivatives(self, data):
        return data.stack([self._normalize_derivatives(d) for d in data])

    def _count_activation(self, layer_input):
        raise NotImplementedError

    def _save_to_file(self, filename, data):
        with open(self.layer_folder + "/" + filename, 'wb') as f:
            pickle.dump(data, f)

    def _load_from_file(self, filename):
        try:
            with open(self.layer_folder + "/" + filename, 'rb') as f:
                data = pickle.load(f)
            return data
        except:
            return None

    def save_activations(self, activations):
        self._save_to_file("activations", activations)

    def load_activations(self):
        return self._load_from_file("activations")

    def save_derivatives(self, derivatives):
        self._save_to_file("derivatives", derivatives)

    def load_derivatives(self):
        return self._load_from_file("derivatives")

    def count_activation(self, layer_input):
        """
        Returns estimated activations

        :param Numlike layer_input:
        :return Numlike: activations
        """
        layer_input = self.normalize_activation(layer_input)
        input_shape = change_order(make_iterable(self.layer.input_shape))
        layer_input = layer_input.reshape(input_shape)
        if self.need_activation:
            self.save_activations(layer_input)
        return self._count_activation(layer_input)

    def _count_derivatives(self, layer_output, input_shape):
        raise NotImplementedError

    def count_derivatives(self, layer_output, batches):
        """
        Returns estimated impact of input of layer on output of network

        :param Numlike layer_output:
        :param tuple input_shape: shape of input
        :param int batches: number of batches
        :return Numlike: derivatives
        """
        layer_output = self.normalize_derivatives(layer_output)
        input_shape = add_tuples(batches,
                                 change_order(self.layer.input_shape))
        output_shape = add_tuples(batches,
                                  change_order(self.layer.output_shape))
        layer_output = layer_output.reshape(output_shape)

        if self.need_derivatives:
            derivatives = self.load_derivatives()
            new_derivatives = layer_output.sum((0,))
            if derivatives is not None:
                derivatives = derivatives + new_derivatives
            else:
                derivatives = new_derivatives
            self.save_derivatives(derivatives)

        return self._count_derivatives(layer_output, input_shape)

    def count_derest(self, count_function):
        """
        Returns indicators of each weight importance

        :param function count_function: function to count indicators,
            takes Numlike and returns float
        :return list of numpy arrays:
        """
        return []
