"""Layer and WeightedLayer."""

import numpy as np

import theano


class Layer(object):
    """Network layer."""
    def __init__(self, input_layer_name=None, name='layer'):
        """Create layer.

        :param input_layer_name: Optional name of input layer. If None, then
                                 :class:`Network` will set preceding layer as
                                 input layer.
        :param name: Optional name of layer. If set, it can later be used as
                     `input_layer_name` for another layer.
        """
        self.output = None
        self.train_output = None
        self.cost = None
        self._input_shape = None
        self._input = None
        self._train_input = None
        self._input_layer = None
        self.batch_size = None

        self.name = name
        self.input_layer_name = input_layer_name

    def _reshape_input(self, raw_layer_input):
        """Return input in the correct format for given layer.

        :param raw_layer_input: Layer input.
        :return: Reshaped input.
        """
        return raw_layer_input

    def _get_output(self, layer_input):
        """Return layer's output.

        :param layer_input: Layer input.
        :return: Layer output.
        """
        return layer_input

    def _get_train_output(self, layer_input):
        """Return layer's output used for training.

        :param layer_input: Layer input.
        :return: Layer train output.
        """
        return self._get_output(layer_input)

    @property
    def input(self):
        """Layer input."""
        return self._input

    @input.setter
    def input(self, value):
        self._input = self._reshape_input(value)
        self.output = self._get_output(self.input)

    @property
    def train_input(self):
        """Layer input used for training."""
        if self._train_input:
            return self._train_input
        return self._input

    @train_input.setter
    def train_input(self, value):
        self._train_input = self._reshape_input(value)
        self.train_output = self._get_train_output(self.train_input)

    @property
    def input_shape(self):
        return self._input_shape

    @input_shape.setter
    def input_shape(self, value):
        self._input_shape = value

    @property
    def output_shape(self):
        return self.input_shape

    @property
    def input_layer(self):
        return self._input_layer

    @input_layer.setter
    def input_layer(self, input_layer):
        self._input_layer = input_layer
        self.input_shape = input_layer.output_shape

        self.input = input_layer.output
        if input_layer.train_output is not None:
            self.train_input = input_layer.train_output

    def set_params(self, params):
        """Set layer's weights and biases, if it has any.

        :param params: Weights and biases. Exact format depends on layer type.
        """
        pass

    def get_params(self, params):
        """Get layer's weights and biases, if it has any.

        :returns: Weights and biases. Exact format depends on layer type.
        """
        pass


class WeightedLayer(Layer):
    """Layer with weights and biases."""
    def __init__(self, input_layer_name=None, name='weight_layer'):
        """Create weighted layer."""
        super(WeightedLayer, self).__init__(input_layer_name, name)
        self.W_shared = None
        self.b_shared = None
        self.W_velocity = None
        self.b_velocity = None

    @property
    def W(self):
        """Copy of layer's weights."""
        return self.W_shared.get_value()

    @W.setter
    def W(self, value):
        self.W_shared.set_value(value)

    @property
    def b(self):
        """Copy of the layer's biases."""
        return self.b_shared.get_value()

    @b.setter
    def b(self, value):
        self.b_shared.set_value(value)

    def alloc_velocity(self):
        """Create velocity tensors for weights and biases.

        Velocity tensors have the same size as corresponding weights and biases
        tensors, so note that creating velocities results in doubling size of
        the layer.
        Velocity tensors should be freed after training to save device memory.
        """
        self.W_velocity = theano.shared(
            np.zeros_like(self.W, dtype=theano.config.floatX),
            borrow=True)
        self.b_velocity = theano.shared(
            np.zeros_like(self.b, dtype=theano.config.floatX),
            borrow=True)

    def free_velocity(self):
        """Remove velocity tensors."""
        self.W_velocity = None
        self.b_velocity = None

    def set_params(self, params):
        self.W, self.b = params

    def get_params(self, params):
        return self.W, self.b
