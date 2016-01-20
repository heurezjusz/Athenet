"""Training LeNet on MNIST data."""

from athenet import Network
from athenet.layers import ReLU, Softmax, MaxPool, FullyConnectedLayer,\
    ConvolutionalLayer
from athenet.utils import BIN_DIR, DATA_DIR, load_data, get_bin_path,\
    save_data_to_pickle, MNISTDataLoader

lenet_filename = 'lenet_weights.pkl.gz'
lenet_url = 'http://students.mimuw.edu.pl/~wg346897/hosting/athenet/lenet_weights.pkl.gz'

def lenet_untrained():
    lenet = Network([
        ConvolutionalLayer(image_shape=(28, 28, 1), filter_shape=(5, 5, 20)),
        ReLU(),
        MaxPool(poolsize=(2, 2)),
        ConvolutionalLayer(filter_shape=(5, 5, 50)),
        ReLU(),
        MaxPool(poolsize=(2, 2)),
        FullyConnectedLayer(n_out=500),
        ReLU(),
        FullyConnectedLayer(n_out=10),
        Softmax(),
    ])
    return lenet

def load_lenet_weights():
    try:
        return load_data(get_bin_path(lenet_filename), lenet_url)
    except:
        return None


def lenet():
    lenet_embryo = lenet_untrained()
    lenet_embryo.data_loader = MNISTDataLoader()
    weights = load_lenet_weights()
    if weights:
        lenet_embryo.set_params(weights)
    else:
        print 'lenet file not found, will be trained now...'
        lenet_embryo.train(learning_rate=0.1, n_epochs=10, batch_size=300)
        filename = get_bin_path(lenet_filename)
        save_data_to_pickle(filename)
    return lenet_embryo
