"""
The cache module handles caching of certain expensive function calls.
"""

from collections import Iterable, Mapping
from functools import wraps

_CACHE = {}


def _hash(thing):
    try:
        return hash(thing)
    except TypeError:
        if isinstance(thing, Mapping):
            return _hash(sorted(thing.items()))
        elif isinstance(thing, Iterable):
            return hash(tuple(_hash(t) for t in thing))
        raise

CacheMiss = object()


def cacheit(func):
    @wraps(func)
    def wrapped_func(*args, **kwargs):
        result = _CACHE.get(_hash((func, args, kwargs)), CacheMiss)
        if result is not CacheMiss:
            #print('CACHE HIT')
            return result
        else:
            #print('CACHE MISS')
            result = func(*args, **kwargs)
            _CACHE[_hash((func, args, kwargs))] = result
            return result
    return wrapped_func
