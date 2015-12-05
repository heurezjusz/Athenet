"""API for Athena, implemented in Theano."""

import timeit

import numpy as np
import theano
import theano.tensor as T


class Network(object):
    """Neural network."""
    def __init__(self, layers, x, y, output_function=lambda y: y):
        """
        Create neural network.

        layers: List of layer from which the network is to be created
        x: Input format
        y: Output format
        output_function: Function mapping from network output to desired
        output
        """
        self.layers = layers
        self.x = x
        self.y = y
        layers[0].set_input(self.x)
        for i in xrange(1, len(self.layers)):
            layer, prev_layer = self.layers[i], self.layers[i-1]
            layer.set_input(prev_layer.output)
        self.output = self.layers[-1].output
        self.y_out = output_function(self.output)

    def accuracy(self, y):
        """Return average network accuracy

        y: List of desired outputs
        return: A number between 0 and 1 representing average accuracy
        """
        return T.mean(T.eq(y, self.y_out))

    def train(self, datasets, learning_rate=0.01, n_epochs=100,
              batch_size=20):
        """Train and test the network

        datasets: Train, validation and test sets
        learning_rate: Learning rate
        n_epochs: Number of epochs
        batch_size: Size of one minibatch
        """
        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]

        n_train_batches =\
            train_set_x.get_value(borrow=True).shape[0] / batch_size
        n_valid_batches =\
            valid_set_x.get_value(borrow=True).shape[0] / batch_size
        n_test_batches =\
            test_set_x.get_value(borrow=True).shape[0] / batch_size

        # set cost function for the last layer
        self.layers[-1].set_cost(self.y)
        cost = self.layers[-1].cost

        weighted_layers = [layer for layer in self.layers
                           if isinstance(layer, WeightedLayer)]
        params = []
        for layer in weighted_layers:
            params += layer.params
        grad = [T.grad(cost, param) for param in params]
        updates = [(param, param - learning_rate*derivative)
                   for param, derivative in zip(params, grad)]

        index = T.lscalar()  # index to a [mini]batch
        train_model = theano.function(
            inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                self.x: train_set_x[index * batch_size:
                                    (index+1) * batch_size],
                self.y: train_set_y[index * batch_size:
                                    (index+1) * batch_size]
            }
        )
        validate_accuracy = theano.function(
            inputs=[index],
            outputs=self.accuracy(self.y),
            givens={
                self.x: valid_set_x[index * batch_size:
                                    (index+1) * batch_size],
                self.y: valid_set_y[index * batch_size:
                                    (index+1) * batch_size]
            }
        )
        test_accuracy = theano.function(
            inputs=[index],
            outputs=self.accuracy(self.y),
            givens={
                self.x: test_set_x[index * batch_size:(index+1) * batch_size],
                self.y: test_set_y[index * batch_size:(index+1) * batch_size]
            }
        )

        patience = 10000
        patience_increase = 2
        validation_interval = min(n_train_batches, patience / 2)
        best_validation_accuracy = 0.0
        epoch = 0
        iteration = 0
        done_looping = False

        start_time = timeit.default_timer()
        while (epoch < n_epochs) and (not done_looping):
            epoch += 1
            print 'Epoch {}'.format(epoch)
            for minibatch_index in xrange(n_train_batches):
                train_model(minibatch_index)
                iteration += 1
                if iteration % validation_interval == 0:
                    validation_accuracies = [validate_accuracy(i) for i
                                             in xrange(n_valid_batches)]
                    validation_accuracy = np.mean(validation_accuracies)
                    print '\tAccuracy on validation data: {:.2f}%'.format(
                        100 * validation_accuracy)
                    if validation_accuracy > best_validation_accuracy:
                        patience = max(patience, iteration * patience_increase)
                        best_validation_accuracy = validation_accuracy

                if patience <= iteration:
                    done_looping = True
                    break
        end_time = timeit.default_timer()

        test_accuracies = [test_accuracy(i) for i in xrange(n_test_batches)]
        test_accuracy = np.mean(test_accuracies)
        print 'Accuracy on test data: {:.2f}%'.format(100 * test_accuracy)
        print 'Training time: {:.1f}s'.format(end_time - start_time)


class Layer(object):
    """Base class for network layer."""
    def __init__(self):
        self.input = None
        self.output = None
        self.cost = None


class Softmax(Layer):
    """Softmax layer."""
    def set_input(self, input):
        """Set layer's input and output variables.

        input: Variable to be set as layer input
        """
        self.input = input
        self.output = T.nnet.softmax(input)

    def set_cost(self, y):
        """
        Set layer's cost variables.

        y: Desired output
        """
        self.cost = T.mean(-T.log(self.output)[T.arange(y.shape[0]), y])


class Activation(Layer):
    """Layer applying activation function to neurons."""
    def __init__(self, activation_function):
        """
        Create new activation layer.

        activation_function: Activation function to be applied
        """
        super(Activation, self).__init__()
        self.activation_function = activation_function

    def set_input(self, input):
        """Set layer's input and output variables.

        input: Variable to be set as layer input
        """
        self.input = input
        self.output = self.activation_function(input)


class ReLU(Activation):
    """Layer applying rectified linear activation function."""
    def __init__(self):
        super(ReLU, self).__init__(relu)


class WeightedLayer(Layer):
    """Layer with weights and biases."""
    def __init__(self):
        super(WeightedLayer, self).__init__()
        self.W = None
        self.b = None


class FullyConnectedLayer(WeightedLayer):
    """Fully connected layer."""
    def __init__(self, n_in, n_out):
        """
        Create fully connected layer.

        n_in: Number of input neurons
        n_out: Number of output neurons
        """
        super(FullyConnectedLayer, self).__init__()
        W_value = np.asarray(
            np.random.normal(
                loc=0.0,
                scale=np.sqrt(1.0 / (n_out)),
                size=(n_in, n_out)),
            dtype=theano.config.floatX
        )
        self.W = theano.shared(value=W_value, name='w', borrow=True)

        b_value = np.zeros((n_out,), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_value, name='b', borrow=True)
        self.params = [self.W, self.b]

    def set_input(self, input):
        """Set layer's input and output variables.

        input: Variable to be set as layer input
        """
        self.input = input
        self.output = T.dot(self.input, self.W) + self.b


def relu(x):
    """Rectified linear activation function

    x: Neuron input
    """
    return T.maximum(0.0, x)
