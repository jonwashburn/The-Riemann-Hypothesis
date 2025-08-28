#!/usr/bin/env python3
# prime_tail.py — unconditional prime-tail budgets for weighted p-adaptive model

import math
from typing import List


def primes_up_to(N: int) -> List[int]:
    N = int(N)
    if N < 2:
        return []
    sieve = bytearray(b"\x01") * (N + 1)
    sieve[:2] = b"\x00\x00"
    p = 2
    while p * p <= N:
        if sieve[p]:
            step = p
            start = p * p
            sieve[start:N + 1:step] = b"\x00" * (((N - start) // step) + 1)
        p += 1
    return [i for i in range(2, N + 1) if sieve[i]]


def S_gamma_leq_Q(gamma: float, Q: int) -> float:
    ps = primes_up_to(Q)
    return sum((p ** (-gamma)) for p in ps)


def E1_approx(y: float) -> float:
    """Exponential integral E1(y) with simple piecewise approximations.
    For y small: series around 0: E1(y) = -γ - ln y + y - y^2/4 + y^3/18 - ...
    For y large: asymptotic: E1(y) ~ e^{-y}/y * (1 - 1/y + 2!/y^2 - 6/y^3).
    """
    if y <= 0:
        # Guard: E1 undefined at y<=0; return large number
        return 1e9
    if y < 0.5:
        gamma_E = 0.5772156649015328606
        # Use first few terms
        t = y
        s = -gamma_E - math.log(y)
        s += t
        t2 = t * t
        s -= t2 / 4.0
        t3 = t2 * t
        s += t3 / 18.0
        t4 = t3 * t
        s -= t4 / 96.0
        return s
    else:
        inv = 1.0 / y
        e = math.exp(-y)
        series = 1.0 - inv + 2.0 * (inv ** 2) - 6.0 * (inv ** 3)
        return e * inv * series


def prime_tail_sum_upper(gamma: float, P: int, Cpi: float = 1.26) -> float:
    """Unconditional upper bound for sum_{p >= P} p^{-gamma}, gamma>1, P>=55.
    Uses pi(x) <= Cpi x/log x and partial summation, yielding:
      sum_{p>=P} p^{-gamma} <= Cpi (gamma E1((gamma-1) log P) + P^{1-gamma}/log P).
    For P<55, includes exact finite head from P up to 55.
    """
    if gamma <= 1.0:
        raise ValueError("gamma must be > 1.")
    if P < 55:
        head_primes = [p for p in primes_up_to(55) if p >= P]
        head = sum((p ** (-gamma)) for p in head_primes)
        return head + prime_tail_sum_upper(gamma, 55, Cpi)
    y = (gamma - 1.0) * math.log(P)
    main = E1_approx(y)
    boundary = (P ** (1.0 - gamma)) / math.log(P)
    return Cpi * (gamma * main + boundary)


def delta_FF(sigma: float, pmin: int, Cwin: float = 0.25, Cpi: float = 1.26) -> float:
    gamma = sigma + 0.5
    tail = prime_tail_sum_upper(gamma, pmin, Cpi)
    return (Cwin / 4.0) * (pmin ** (-gamma)) * tail


def delta_FS_far_row(sigma: float, Q: int, pmin: int, Cwin: float = 0.25) -> float:
    gamma = sigma + 0.5
    Sg = S_gamma_leq_Q(gamma, Q)
    return (Cwin / 4.0) * (pmin ** (-gamma)) * Sg


def delta_SF_small_row(sigma: float, Q: int, pmin: int, Cwin: float = 0.25, Cpi: float = 1.26) -> float:
    gamma = sigma + 0.5
    tail = prime_tail_sum_upper(gamma, pmin, Cpi)
    worst_small = 2 ** (-gamma) if Q >= 2 else (3 ** (-gamma))
    return (Cwin / 4.0) * worst_small * tail


def delta_SS(sigma: float, Q: int, Cwin: float = 0.25) -> float:
    gamma = sigma + 0.5
    primes = primes_up_to(Q)
    Sg = sum((p ** (-gamma)) for p in primes)
    worst = 0.0
    for p in primes:
        w = (p ** (-gamma)) * (Sg - (p ** (-gamma)))
        if w > worst:
            worst = w
    return (Cwin / 4.0) * worst


def choose_pmin_for_targets(sigma: float, Q: int, tau_FF: float, tau_FS: float,
                            pmin_cap: int = 10**6, Cwin: float = 0.25, Cpi: float = 1.26) -> int:
    p = max(55, Q + 2)
    while p <= pmin_cap:
        if delta_FF(sigma, p, Cwin, Cpi) <= tau_FF and delta_FS_far_row(sigma, Q, p, Cwin) <= tau_FS:
            return p
        # accelerate scan
        p = int(p * 1.05) + 1
    return pmin_cap + 1


