from string import *
from functools import *
from inspect import getargspec
from itertools import *
from operator import *

alph = ascii_lowercase
ALPH = ascii_uppercase
Alph = ascii_letters

num = digits

p = product
l = list

def curried(n):
    def curry(fn):
        def _inner(*args):
            if len(args) < n:
                return curried(n - len(args))(partial(fn, *args))
            return fn(*args)
        return _inner
    return curry

@curried(2)
def m(*args, **kwargs):
    return list(map(*args, **kwargs))

@curried(2)
def f(*args, **kwargs):
    return list(filter(*args, **kwargs))

@curried(2)
def member(objs, obj):
    return contains(objs, obj)
