from typing import Any, List


def rotate_list(l: List[Any], n: int):
    """Helper function for rotating lists, typically list of Players

    Parameters
    ----------
    l : List[Any]
        List to rotate.
    n : int
        Integer index of where to rotate.
    """
    if n > len(l):
        raise ValueError
    return l[n:] + l[:n]
