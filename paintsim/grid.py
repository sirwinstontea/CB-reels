"""Uniform-grid neighbour search via spatial hashing (cell size = h).

Each particle stores its integer cell, so two cells that collide in the hash table never produce duplicate
or false neighbours.
"""
import numpy as np
from numba import njit, prange

_P1, _P2, _P3 = 73856093, 19349663, 83492791


def table_size(n):
    t = 1024
    while t < 2 * max(n, 1):
        t <<= 1
    return t


@njit(cache=True)
def build_hash(x, h, T):
    n = x.shape[0]
    inv = 1.0 / h
    pc = np.empty((n, 3), np.int64)
    key = np.empty(n, np.int64)
    start = np.zeros(T + 1, np.int64)
    for i in range(n):
        cx = int(np.floor(x[i, 0] * inv))
        cy = int(np.floor(x[i, 1] * inv))
        cz = int(np.floor(x[i, 2] * inv))
        pc[i, 0] = cx
        pc[i, 1] = cy
        pc[i, 2] = cz
        k = ((cx * _P1) ^ (cy * _P2) ^ (cz * _P3)) % T
        key[i] = k
        start[k + 1] += 1
    for k in range(T):
        start[k + 1] += start[k]
    fill = start[:-1].copy()
    order = np.empty(n, np.int64)
    for i in range(n):
        order[fill[key[i]]] = i
        fill[key[i]] += 1
    return start, order, pc


@njit(parallel=True, cache=True)
def query(xq, xd, start, order, pcd, h, T, maxn, same_set):
    """Neighbours in data set xd within distance h of each query point. same_set=True skips j == i."""
    nq = xq.shape[0]
    nbr = np.empty((nq, maxn), np.int32)
    cnt = np.zeros(nq, np.int32)
    over = np.zeros(nq, np.int32)
    h2 = h * h
    inv = 1.0 / h
    for i in prange(nq):
        cx = int(np.floor(xq[i, 0] * inv))
        cy = int(np.floor(xq[i, 1] * inv))
        cz = int(np.floor(xq[i, 2] * inv))
        c = 0
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    ncx = cx + dx
                    ncy = cy + dy
                    ncz = cz + dz
                    k = ((ncx * _P1) ^ (ncy * _P2) ^ (ncz * _P3)) % T
                    for s in range(start[k], start[k + 1]):
                        j = order[s]
                        if pcd[j, 0] != ncx or pcd[j, 1] != ncy or pcd[j, 2] != ncz:
                            continue
                        if same_set and j == i:
                            continue
                        rx = xq[i, 0] - xd[j, 0]
                        ry = xq[i, 1] - xd[j, 1]
                        rz = xq[i, 2] - xd[j, 2]
                        if rx * rx + ry * ry + rz * rz < h2:
                            if c < maxn:
                                nbr[i, c] = j
                                c += 1
                            else:
                                over[i] = 1
        cnt[i] = c
    return nbr, cnt, over.sum()


class HashGrid:
    """Static or per-step hash of a point set."""

    def __init__(self, x, h):
        self.h = h
        self.T = table_size(x.shape[0])
        self.x = x
        self.start, self.order, self.pc = build_hash(x, h, self.T)

    def neighbors(self, xq, maxn, same_set=False):
        return query(xq, self.x, self.start, self.order, self.pc, self.h, self.T, maxn, same_set)
