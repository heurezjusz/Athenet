from src.sparsifying.sparsify_smallest import rat1_on_network

def sparsify(network, algorithm='rat1', percentage = 7, *args):
    if algorithm == 'rat1':
        rat1_on_network(network, percentage)
    
    return network