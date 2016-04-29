# from athenet.network import Network
from athenet.layers import InceptionLayer, ConvolutionalLayer
import theano
from athenet.algorithm.derest.network import get_derest_layer
from athenet.algorithm.numlike import Nplike, Interval
# from athenet.algorithm.numlike.interval import Interval
import theano.tensor as T

import numpy as np

def eval_tensor_on_layer(layer, tensor):
    layer.input = T.tensor4('tmp')
    evaluate = theano.function(
        [layer.input_layer.input],
        layer.output
    )
    return evaluate(tensor)


def eval_tensor_on_layers(layer1, layer2, tensor):
    layer1.input = T.tensor4()
    layer2.input_layer = layer1
    evaluate = theano.function(
        [layer1.input],
        layer2.output
    )
    return evaluate(tensor)


def testa():
    dummy_layer = ConvolutionalLayer((1,1,1),(10,10,1))
    input = np.ones((1, 1, 10, 10), dtype=theano.config.floatX)
    dummy_layer.input = input
    layer = InceptionLayer([1,2,3,4,5,6])
    layer.input_layer = dummy_layer

    print "Layer exists"
    print eval_tensor_on_layers(dummy_layer, layer, input)

    print "Layer evaluated"
    print layer.output_shape
    print layer.input_shape
    derest_layer = get_derest_layer(layer)
    print "Derest layer exist"
    act = Interval(np.zeros((1,10,10)), np.ones((1,10,10)))

    a = T.tensor3(name = "llower", dtype=theano.config.floatX)
    b = T.tensor3(name="lupper", dtype=theano.config.floatX)
    act_in_theory = Interval(a,b)
    out_in_theory = derest_layer.count_activation(act_in_theory)
    #foo = theano.function([a,b], out_in_theory.eval())
    print "run!"
    res = [r.eval({b: act.upper, a:act.lower}) for r in out_in_theory]
    print res
    print layer.output_shape


def test_numlike():
    a = np.ones((3,4,5))
    b = 2 * a
    na = Nplike(a)
    nb = Nplike(b)
    print na.concat(nb).eval()
    ia = Interval(a, a * 3)
    ib = Interval(b, b * 3)

    print ia

    print ia.concat(ib).eval()

testa()