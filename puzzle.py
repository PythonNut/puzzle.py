from string import *
from functools import *
from inspect import getargspec
from itertools import *
from operator import *
import numpy as np

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

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def int2base(x, base, digs=digits + ascii_letters):
    if x == 0: return digs[0]

    sign, digits = np.sign(x), []
    x *= sign

    while x:
        digits.append(digs[int(x % base)])
        x = int(x / base)

    if sign < 0: digits.append('-')
    return ''.join(reversed(digits))

def is_palindrome(s, err=0):
    penalty = 0
    for a, b in zip(s, reversed(s)):
        if a != b:
            penalty += 1
    return penalty <= 2 * err

def substrings(s):
    return (s[i:j+1] for i in range(len(s)) for j in range(i,len(s)))
