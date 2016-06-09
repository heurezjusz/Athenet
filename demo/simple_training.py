import athenet
from athenet.data_loader import MNISTDataLoader
from athenet import Network
from athenet.layers import FullyConnectedLayer, ReLU, Softmax


network = Network([
    FullyConnectedLayer(n_in=28*28, n_out=1000),
    ReLU(),
    FullyConnectedLayer(n_out=800),
    ReLU(),
    FullyConnectedLayer(n_out=500),
    ReLU(),
    FullyConnectedLayer(n_out=10),
    Softmax(),
])
network.data_loader = MNISTDataLoader()

config = athenet.TrainConfig()
config.n_epochs = 10
config.batch_size = 300
config.learning_rate = 0.1

network.train(config)
print 'Accuracy on test data: {:.2f}%'.format(100*network.test_accuracy())






