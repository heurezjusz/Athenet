import numpy as np


def list_of_percentage_rows_table(table, layer_id):
    """returns list of tuples (percentage, number of row, layer_id)"""
    all_weights = np.sum(abs(table))
    result = []
    for i in xrange(table.shape[0]):
        result.append((np.sum(abs(table[i])) / all_weights, i, layer_id))
    return result


def list_of_percentage_columns(layer_id, layer):
    return list_of_percentage_rows_table(np.transpose(layer.W), layer_id)


def list_of_percentage_rows(layer_id, layer):
    return list_of_percentage_rows_table(layer.W, layer_id)


def delete_column(layer, i):
    W = layer.W
    print "delete column", i, "from table of shape", W.shape
    for j in xrange(W.shape[0]):
        W[j][i] = 0.
    layer.W = W


def delete_row(layer, i):
    W = layer.W
    print "delete row", i, "from table of shape", W.shape
    for j in xrange(W.shape[1]):
        W[i][j] = 0.
    layer.W = W
