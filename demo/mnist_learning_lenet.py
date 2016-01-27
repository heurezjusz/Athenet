"""Training LeNet on MNIST data."""

import athenet
from athenet.utils import MNISTDataLoader


network = athenet.models.lenet(trained=False)
network.data_loader = MNISTDataLoader()

network.train(learning_rate=0.1, n_epochs=10, batch_size=300)
