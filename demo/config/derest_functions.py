import numpy

""" normalization functions """

def no_normalization(data):
    return data


def normalization_by_length(data, max_length=1.):
    a = data.sum()
    return max_length * data / (a.upper - a.lower + 1e-6)


def normalization_by_max_value(data, max_value=1.):
    a = data.abs().amax()
    return max_value * data / (a.upper + 1e-6)


""" indicators coumputing functions """


def sum_max(values):
    return values.sum().abs().upper


def sum_length(values):
    a = values.sum()
    return a.upper - a.lower


def max_length(values):
    a = values.amax().upper
    b = - values.neg().amax().upper
    return a - b