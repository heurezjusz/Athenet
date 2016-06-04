"""
Custom functions for derest algorithm
"""

""" normalization functions """


def no_normalization(data):
    return data


def normalization_by_length(data, max_length=1.):
    a = data.sum()
    return max_length * data / (a.upper - a.lower + 1e-6)


def normalization_by_max_value(data, max_value=1.):
    a = data.abs().amax()
    return max_value * data / (a.upper + 1e-6)

derest_normalization = {
    "none": no_normalization,
    "length": normalization_by_length,
    "max_value": normalization_by_max_value
}

""" indicators coumputing functions """


def count_max(value):
    return value.abs().upper


def count_length(value):
    return value.upper - value.lower


derest_indicators = {
    "max": count_max,
    "length": count_length
}


def get_derest_params(args):
    kwargs = dict()
    kwargs["max_batch_size"] = args.batch_size if args.batch_size > 0 else None
    if args.normalize_activations != "default":
        kwargs["normalize_activations"] = \
            derest_normalization[args.normalize_activations]
    if args.normalize_derivatives != "default":
        kwargs["normalize_derivatives"] = \
            derest_normalization[args.normalize_derivatives]
    if args.derest_count_function != "default":
        kwargs["count_function"] = \
            derest_indicators[args.derest_count_function]

    return kwargs
