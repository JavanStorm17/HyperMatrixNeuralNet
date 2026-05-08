"""Tiny pure-Python numeric helpers. No numpy dependency on purpose."""

import math
import random


def vzeros(n):
    return [0.0] * n


def vrand(n, scale=0.1):
    return [random.gauss(0.0, scale) for _ in range(n)]


def vadd(a, b):
    return [x + y for x, y in zip(a, b)]


def vsub(a, b):
    return [x - y for x, y in zip(a, b)]


def vscale(a, s):
    return [x * s for x in a]


def vdot(a, b):
    return sum(x * y for x, y in zip(a, b))


def vnorm(a):
    n = math.sqrt(sum(x * x for x in a)) + 1e-9
    return [x / n for x in a]


def cosine(a, b):
    if not a or not b:
        return 0.0
    na = math.sqrt(sum(x * x for x in a)) + 1e-9
    nb = math.sqrt(sum(x * x for x in b)) + 1e-9
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def matvec(W, x):
    return [vdot(row, x) for row in W]


def matrand(rows, cols, scale=0.1):
    return [vrand(cols, scale) for _ in range(rows)]


def relu(v):
    return [x if x > 0.0 else 0.0 for x in v]


def tanh(v):
    return [math.tanh(x) for x in v]


def sigmoid(v):
    return [1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x)))) for x in v]


def topk_mask(v, k):
    """Return v with all but the top-k absolute values zeroed (sparse activation)."""
    if k >= len(v):
        return list(v)
    idx = sorted(range(len(v)), key=lambda i: -abs(v[i]))[:k]
    keep = set(idx)
    return [v[i] if i in keep else 0.0 for i in range(len(v))]


def project(x, dim_out):
    """Pad or truncate a vector to dim_out. Used for crude inter-dim adapting."""
    if len(x) == dim_out:
        return list(x)
    if len(x) > dim_out:
        return x[:dim_out]
    return list(x) + [0.0] * (dim_out - len(x))
