"""Auxiliary functions for derest."""

def assert_numlike(value):
    if not isinstance(value, Numlike):
        raise ValueError("layer_input must be Numlike.")
