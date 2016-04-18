
def _change_order(a):
    """
    So the last will be first
    """
    try:
        h, w, n = a
        return (n, h, w)
    except:
        return a