import numpy as np

def set_zeros_on_layer(layer, percentage, order):
    W = layer.W
    percentile = np.percentile([order(a) for a in W.flat], percentage)
    W[order(W) < percentile] = 0
    layer.W = W   
    
def set_zeros_on_network(network, percentage, order):
    weights = np.concatenate([layer.W.flatten() for layer in network.weighted_layers])
    percentile = np.percentile([order(a) for a in weights.flat], percentage)
    for layer in network.weighted_layers:
        W = layer.W
        W[order(W) < percentile] = 0
        layer.W = W 
       
def sparsify_smallest_on_network(network, percentage):
    set_zeros_on_network(network, percentage, abs)
    
def sparsify_nearest_to_network_mean(network, percentage):
    weights = np.concatenate([layer.W.flatten() for layer in network.weighted_layers])
    mean = np.mean(weights)
    set_zeros_on_network(network, percentage, lambda x: abs(mean - x))
    
def sparsify_smallest_on_layers(network, percentage):
    for layer in network.weighted_layers:
        set_zeros_on_layer(layer, percentage, abs)
        
def sparsify_nearest_to_layer_mean(network, percentage):
    for layer in network.weighted_layers:
        mean = np.mean(layer.W.flatten())
        set_zeros_on_layer(layer, percentage, lambda x: abs(mean - x))
        
    
        