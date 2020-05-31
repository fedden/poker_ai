from typing import Any, List


def rotate_list(l: List[Any], n: int):
    if n > len(l):
        raise ValueError
    return l[n:] + l[:n]
