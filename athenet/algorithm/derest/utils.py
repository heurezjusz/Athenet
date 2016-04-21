def change_order(a):
    """
    So the last will be first
    """
    try:
        h, w, n = a
        return (n, h, w)
    except:
        return a


def add_tuples(a, b):
    if not isinstance(a, tuple):
        a = (a, )
    if not isinstance(b, tuple):
        b = (b, )
    return a + b


