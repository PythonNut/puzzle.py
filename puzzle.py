from string import *
from functools import *
from inspect import getargspec
from itertools import *
from operator import *
from collections import *
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

    return "".join(result)


def apply(f, *args, **kwargs):
    return f(*args, **kwargs)


def decode_nums_alphabetic(nums, start=1):
    return "".join(modidx(alph, n - start) for n in nums)


def decode_nums_ascii(nums):
    return "".join(chr, nums)


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def int2base(x, base, digs=digits + ascii_letters):
    if x == 0:
        return digs[0]

    sign, digits = np.sign(x), []
    x *= sign

    while x:
        digits.append(digs[int(x % base)])
        x = int(x / base)

    if sign < 0:
        digits.append("-")
    return "".join(reversed(digits))


def is_palindrome(s, err=0):
    penalty = 0
    for a, b in zip(s, reversed(s)):
        if a != b:
            penalty += 1
    return penalty <= 2 * err


def substrings(s):
    return (s[i : j + 1] for i in range(len(s)) for j in range(i, len(s)))


def chunks(l, n, overlap=False):
    for i in range(0, len(l) - n + 1, 1 if overlap else n):
        yield l[i : i + n]


def subsequences(l):
    for length in range(len(l), 0, -1):
        for combination in combinations(l, length):
            yield combination


class IndexConvertor(object):
    def __init__(self, shape, offset):
        self.shape = shape
        self.offset = offset

    def __getitem__(self, i):
        if len(self.shape) == 1:
            assert 0 <= i < self.shape[0]
            return i + self.offset

        return int(np.ravel_multi_index(i, self.shape) + self.offset)

    def __call__(self, j):
        return np.unravel_index(j - self.offset, self.shape)

    def __str__(self):
        return f"I{self.shape}"

    __repr__ = __str__

    def size(self):
        return reduce(op.mul, self.shape)


class VariableDispenser(object):
    def __init__(self):
        self.offset = 1
        self.variables = []

    def dispense(self, shape):
        I = IndexConvertor(shape, self.offset)
        self.offset += I.size()
        self.variables.append(I)
        return I

    def __str__(self):
        return f"VD{self.variables}"

    __repr__ = __str__

    def size(self):
        return self.offset - 1

    def top_var(self):
        return self.size()

    def unflatten(self, flat):
        assert len(flat) <= self.size()
        results = []
        i = 0
        for v in self.variables:
            A = np.zeros(v.shape, np.bool)
            for _ in range(v.size()):
                i += 1
                j = v(i)
                if i > len(flat):
                    A[j] = False
                else:
                    A[v(i)] = flat[i - 1] > 0
            results.append(A)

        return results

EIGHT_DIRECTIONS = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
FOUR_DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]

def word_search(words, arr, directions=EIGHT_DIRECTIONS):
    P = product(
        range(len(arr)),
        range(len(arr[0])),
        words,
        directions
    )
    results = defaultdict(lambda: [])
    for i, j, word, (di, dj) in P:
        try:
            idxs = [(i+di*k, j+dj*k) for k in range(len(word))]
            test = ''.join([arr[a][b] for (a, b) in idxs])
            if test == word:
                results[word].append(idxs)
        except IndexError: pass

    return results
