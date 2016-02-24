"""Training LeNet on MNIST data."""

import athenet
from athenet import TrainConfig, ValUnits
from athenet.data_loader import MNISTDataLoader


network = athenet.models.lenet(trained=False)
network.data_loader = MNISTDataLoader()

config = TrainConfig()

config.n_epochs = 10
config.batch_size = 300
config.learning_rate = 0.1
config.val_interval = 1
config.val_units = ValUnits.epochs

network.train(config)
print 'Accuracy on test data: {:.2f}%'.format(100*network.test_accuracy())
