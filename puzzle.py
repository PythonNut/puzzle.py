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

def modidx(objs, idx):
    return objs[idx % len(objs)]

@curried(2)
def m(*args, **kwargs):
    return list(map(*args, **kwargs))

@curried(2)
def f(*args, **kwargs):
    return list(filter(*args, **kwargs))

@curried(2)
def member(objs, obj):
    return contains(objs, obj)

@curried(2)
def rot(n, string, charset=alph):
    result = []
    for char in string:
        if char in charset:
            result.append(modidx(charset, charset.index(char) + n))
        else:
            result.append(char)

    return ''.join(result)

def apply(f, *args, **kwargs): return f(*args, **kwargs)

def decode_nums_alphabetic(nums, start=1):
    return ''.join(modidx(alph, n-start) for n in nums)

def decode_nums_ascii(nums):
    return ''.join(chr, nums)
