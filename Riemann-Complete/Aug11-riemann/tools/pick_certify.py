#!/usr/bin/env python3
"""
Pick-matrix certification of (P+) on boundary intervals for
Theta = (2*J - 1)/(2*J + 1), J = det2(I - A)/(O * xi).

This driver uses a practical surrogate with O(s) = 1 by default:
    J0(s) = det2(I - A(s)) / xi(s),
where det2(I - A) = prod_p (1 - p^{-s}) * exp(p^{-s}) truncated to primes <= Pmax.

NOTE: To certify (P+) rigorously, supply an implementation of O(s) built from
uniform L1 boundary control (outer normalization). Replace J(s) accordingly.
"""
from __future__ import annotations

import math
import json
import argparse
import importlib
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Sequence, Tuple, Optional, Dict, Callable

import numpy as np

try:
    import mpmath as mp
except Exception as e:
    raise SystemExit("This script requires mpmath. Install with: pip install mpmath")


# ------------------------------
# Prime utilities
# ------------------------------
def primes_upto(n: int) -> List[int]:
    if n < 2:
        return []
    sieve = bytearray(b"\x01") * (n + 1)
    sieve[0:2] = b"\x00\x00"
    r = int(n ** 0.5)
    for p in range(2, r + 1):
        if sieve[p]:
            step = p
            start = p * p
            sieve[start:n + 1:step] = b"\x00" * ((n - start) // step + 1)
    return [i for i, v in enumerate(sieve) if v]


# ------------------------------
# Special functions: xi and det2 surrogate
# ------------------------------
def xi(s: complex) -> complex:
    z = mp.mpc(s.real, s.imag)
    return 0.5 * z * (z - 1) * mp.power(mp.pi, -z / 2) * mp.gamma(z / 2) * mp.zeta(z)


def det2_I_minus_A_product(s: complex, primes: Sequence[int]) -> complex:
    z = mp.mpc(s.real, s.imag)
    prod = mp.mpc(1.0)
    for p in primes:
        ps = mp.power(p, -z)
        prod *= (1.0 - ps) * mp.e**(ps)
    return prod


def det2_I_minus_A_series(s: complex, primes: Sequence[int], kmax: int = 12) -> complex:
    # log det2 = - sum_{k>=2} (1/k) sum_p p^{-k s}
    z = mp.mpc(s.real, s.imag)
    S = mp.mpf('0')
    for k in range(2, kmax + 1):
        inner = mp.mpf('0')
        for p in primes:
            inner += mp.power(p, -k * z)
        S -= inner / k
    return mp.e**(S)

# Global det2 selector
DET2_FUNC: Callable[[complex, Sequence[int]], complex] = det2_I_minus_A_product
DET2_KMAX: int = 12


# ------------------------------
# Outer builders
# ------------------------------
class Outer:
    def __call__(self, s: complex) -> complex:
        return 1.0  # placeholder; replace with outer built from boundary L1 data


@dataclass
class OuterCallable(Outer):
    fn: Callable[[complex], complex]
    def __call__(self, s: complex) -> complex:
        return complex(self.fn(complex(s)))


def load_outer_from_module(module_name: str, func_name: str) -> Outer:
    mod = importlib.import_module(module_name)
    fn = getattr(mod, func_name)
    return OuterCallable(fn)


def uniform_grid(T1: float, T2: float, n: int) -> np.ndarray:
    return np.linspace(T1, T2, n)


def simpson_weights(n: int) -> np.ndarray:
    # n must be odd for classical Simpson; if even, fallback to trapezoidal at the end
    if n < 2:
        return np.ones(n)
    w = np.zeros(n)
    if n % 2 == 1:
        w[0] = 1
        w[-1] = 1
        w[1:-1:2] = 4
        w[2:-1:2] = 2
        w = w / 3.0
    else:
        # composite Simpson on n-1 points + trapezoid on last interval
        m = n - 1
        w[:m] = simpson_weights(m)
        w[-2] *= 1.5  # adjust because last interval uses trapezoid
        w[-1] = 0.5
    return w


class OuterPoisson(Outer):
    """Outer factor via Poisson integral on the half-plane.

    Builds O(s) = exp(U_sigma(t) + i V_sigma(t)) from boundary data u(t) on a grid.
    """
    def __init__(self, t_grid: np.ndarray, u_vals: np.ndarray, sigma: float, weights: Optional[np.ndarray] = None):
        assert t_grid.ndim == 1 and u_vals.ndim == 1 and len(t_grid) == len(u_vals)
        self.t_grid = np.asarray(t_grid, dtype=float)
        self.u_vals = np.asarray(u_vals, dtype=float)
        self.sigma = float(sigma)
        if weights is None:
            # Default: Simpson if uniform grid length is odd; else trapezoidal
            if np.allclose(np.diff(self.t_grid), np.diff(self.t_grid)[0]):
                n = len(self.t_grid)
                w = simpson_weights(n)
                dt = (self.t_grid[-1] - self.t_grid[0]) / (n - 1) if n > 1 else 1.0
                weights = w * dt
            else:
                # trapezoidal on nonuniform grid
                w = np.zeros_like(self.t_grid)
                if len(w) > 1:
                    dt = np.diff(self.t_grid)
                    w[0] = dt[0] / 2.0
                    w[-1] = dt[-1] / 2.0
                    w[1:-1] = (dt[:-1] + dt[1:]) / 2.0
                else:
                    w[0] = 1.0
                weights = w
        self.weights = np.asarray(weights, dtype=float)

    def __call__(self, s: complex) -> complex:
        z = complex(s)
        sigma = z.real - 0.5
        t = z.imag
        # Evaluate Poisson (P) and conjugate (Q) kernels at (sigma, t) against u_vals
        x = t - self.t_grid
        denom = sigma * sigma + x * x
        P = sigma / denom
        Q = x / denom
        U = (1.0 / math.pi) * np.sum(self.u_vals * P * self.weights)
        V = (1.0 / math.pi) * np.sum(self.u_vals * Q * self.weights)
        return complex(mp.e**(U)) * complex(math.cos(V), math.sin(V))


def build_outer_for_sigma(
    T1: float,
    T2: float,
    sigma: float,
    primes: Sequence[int],
    eps_factors: Sequence[float],
    n_outer: int = 801,
    pad: Optional[float] = None,
) -> Outer:
    # Build on expanded interval for numerical stability
    pad = max(10.0, 0.5 * (T2 - T1)) if pad is None else pad
    Tlo, Thi = T1 - pad, T2 + pad
    # Uniform grid for Simpson
    if n_outer % 2 == 0:
        n_outer += 1  # Simpson prefers odd
    t_outer = uniform_grid(Tlo, Thi, n_outer)
    # Aggregate u_eps over a small schedule and average
    u_accum = np.zeros_like(t_outer)
    count = 0
    for f in eps_factors:
        eps = sigma * f
        if eps <= 0:
            continue
        vals = []
        for tt in t_outer:
            s_b = 0.5 + eps + 1j * tt
            val = DET2_FUNC(s_b, primes) / xi(s_b)
            vals.append(float(mp.log(abs(val))))
        u_accum += np.array(vals, dtype=float)
        count += 1
    u_avg = u_accum / max(count, 1)
    weights = simpson_weights(len(t_outer))
    dt = (t_outer[-1] - t_outer[0]) / (len(t_outer) - 1) if len(t_outer) > 1 else 1.0
    weights = weights * dt
    return OuterPoisson(t_outer, u_avg, sigma, weights=weights)


def build_outer_global(
    Tlo: float,
    Thi: float,
    sigmas: Sequence[float],
    primes: Sequence[int],
    eps_factors: Sequence[float],
    n_outer: int = 801,
) -> Dict[float, Outer]:
    cache: Dict[float, Outer] = {}
    for sigma in sigmas:
        # Build over one global span [Tlo,Thi] to reuse across all intervals
        if n_outer % 2 == 0:
            n_outer += 1
        t_outer = uniform_grid(Tlo, Thi, n_outer)
        u_accum = np.zeros_like(t_outer)
        count = 0
        for f in eps_factors:
            eps = sigma * f
            if eps <= 0:
                continue
            vals = []
            for tt in t_outer:
                s_b = 0.5 + eps + 1j * tt
                val = DET2_FUNC(s_b, primes) / xi(s_b)
                vals.append(float(mp.log(abs(val))))
            u_accum += np.array(vals, dtype=float)
            count += 1
        u_avg = u_accum / max(count, 1)
        weights = simpson_weights(len(t_outer))
        dt = (t_outer[-1] - t_outer[0]) / (len(t_outer) - 1) if len(t_outer) > 1 else 1.0
        weights = weights * dt
        cache[sigma] = OuterPoisson(t_outer, u_avg, sigma, weights=weights)
    return cache


# ------------------------------
# J, Theta and Pick matrix
# ------------------------------
def J(s: complex, primes: Sequence[int], outer: Outer | None = None) -> complex:
    if outer is None:
        outer = Outer()
    return DET2_FUNC(s, primes) / (outer(s) * xi(s))


def Theta(s: complex, primes: Sequence[int], outer: Outer | None = None) -> complex:
    z = 2.0 * J(s, primes, outer)
    return (z - 1.0) / (z + 1.0)


def pick_matrix(theta_vals: np.ndarray, s_vals: np.ndarray) -> np.ndarray:
    n = len(s_vals)
    K = np.zeros((n, n), dtype=complex)
    for j in range(n):
        for k in range(n):
            num = 1.0 - theta_vals[j] * np.conjugate(theta_vals[k])
            den = s_vals[j] + np.conjugate(s_vals[k]) - 1.0
            K[j, k] = num / den
    return 0.5 * (K + K.conj().T)


def cheb_grid(T1: float, T2: float, n: int) -> np.ndarray:
    k = np.arange(n)
    x = np.cos(np.pi * (2 * k + 1) / (2 * n))
    return (T1 + T2) / 2.0 + (T2 - T1) * x / 2.0


# ------------------------------
# NP factorization helpers
# ------------------------------

def herglotz_matrix(J_vals: np.ndarray, s_vals: np.ndarray) -> np.ndarray:
    """Build the Herglotz kernel matrix H_J with entries
        H_J(s,t) = (J(s) + \overline{J(t)}) / (s + \overline{t} - 1).

    This kernel is PSD iff J is Herglotz on the half-plane Re s > 1/2.
    """
    n = len(s_vals)
    H = np.zeros((n, n), dtype=complex)
    for j in range(n):
        for k in range(n):
            num = J_vals[j] + np.conjugate(J_vals[k])
            den = s_vals[j] + np.conjugate(s_vals[k]) - 1.0
            H[j, k] = num / den
    return 0.5 * (H + H.conj().T)


def pick_matrix_from_J(J_vals: np.ndarray, s_vals: np.ndarray) -> np.ndarray:
    """Reconstruct the Pick kernel K_Theta from J via the exact factorization
        K_Theta(s,t) = d(s) * H_J(s,t) * \overline{d(t)},
        where d(s) = 2 / (2 J(s) + 1).
    """
    H = herglotz_matrix(J_vals, s_vals)
    d = np.array([2.0 / (2.0 * J_vals[j] + 1.0) for j in range(len(s_vals))], dtype=complex)
    D = np.diag(d)
    K = D @ H @ D.conj().T
    return 0.5 * (K + K.conj().T)


# ------------------------------
# Phase-cone scan (Hilbert transform route)
# ------------------------------

def hilbert_transform_periodic(y: np.ndarray) -> np.ndarray:
    N = len(y)
    Y = np.fft.fft(y)
    H = np.zeros(N)
    if N % 2 == 0:
        H[0] = 0.0
        H[N // 2] = 0.0
        H[1:N // 2] = 2.0
    else:
        H[0] = 0.0
        H[1:(N + 1) // 2] = 2.0
    ya = np.fft.ifft(Y * H)
    return np.imag(ya)


def phase_cone_metrics(
    T1: float,
    T2: float,
    sigma: float,
    primes: Sequence[int],
    outer: Optional[Outer] = None,
    n_phase: int = 4096,
    pad: float = 20.0,
) -> Dict[str, float]:
    # Build uniform padded grid
    Tlo = T1 - pad
    Thi = T2 + pad
    t = uniform_grid(Tlo, Thi, n_phase)
    s_vals = 0.5 + sigma + 1j * t
    # Values
    det_vals = np.array([DET2_FUNC(complex(s), primes) for s in s_vals], dtype=complex)
    xi_vals = np.array([xi(complex(s)) for s in s_vals], dtype=complex)
    if outer is None:
        O_vals = np.ones_like(det_vals, dtype=complex)
    else:
        O_vals = np.array([outer(complex(s)) for s in s_vals], dtype=complex)
    F_vals = det_vals / (O_vals * xi_vals)
    u = np.log(np.abs(F_vals) + 0.0)
    Hu = hilbert_transform_periodic(u)
    # Unwrapped phases
    arg_det = np.unwrap(np.angle(det_vals))
    arg_xi = np.unwrap(np.angle(xi_vals))
    w = arg_det - arg_xi - Hu
    # Restrict to interior window indices
    idx = (t >= T1) & (t <= T2)
    wI = w[idx]
    maxabs = float(np.max(np.abs(wI))) if wI.size else float('nan')
    margin = float(0.5 * math.pi - maxabs) if not math.isnan(maxabs) else float('nan')
    return {"max_abs_w": maxabs, "margin": margin}


def scan_phase_cover(
    intervals: Sequence[Tuple[float, float]],
    sigma: float,
    primes: Sequence[int],
    outer: Optional[Outer] = None,
    n_phase: int = 4096,
    pad: float = 20.0,
    outfile: Optional[str] = None,
) -> List[dict]:
    res = []
    for (T1, T2) in intervals:
        m = phase_cone_metrics(T1, T2, sigma, primes, outer=outer, n_phase=n_phase, pad=pad)
        rec = {"interval": (T1, T2), **m}
        res.append(rec)
        print(f"I=[{T1},{T2}] -> max|w|={m['max_abs_w']:.6f}, margin={m['margin']:.6f}")
    if outfile:
        with open(outfile, "a") as f:
            f.write(json.dumps({"meta": {"timestamp": datetime.utcnow().isoformat() + 'Z', "sigma": sigma}}) + "\n")
            for r in res:
                f.write(json.dumps(r) + "\n")
    return res


# ------------------------------
# Certification routines
# ------------------------------

def certify_interval(
    T1: float,
    T2: float,
    primes: Sequence[int],
    outer: Outer | None = None,
    build_outer: bool = False,
    eps_factors: Sequence[float] = (1.0, 0.5, 0.25),
    sigmas: Sequence[float] = (2e-2, 1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4),
    ns: Sequence[int] = (64, 96, 128, 192, 256, 320, 384),
    tol: float = 1e-10,
    outer_per_sigma: Optional[Dict[float, Outer]] = None,
) -> Tuple[bool, List[float], List[float]]:
    minima_pick: List[float] = []
    # For diagnostics: record Herglotz kernel least eigenvalues to make the obstruction explicit
    herglotz_minima: List[float] = []
    for sigma, n in zip(sigmas, ns):
        local_outer = outer
        if outer_per_sigma is not None and sigma in outer_per_sigma:
            local_outer = outer_per_sigma[sigma]
        elif build_outer:
            local_outer = build_outer_for_sigma(T1, T2, sigma, primes, eps_factors, n_outer=801)
        t_grid = np.sort(cheb_grid(T1, T2, n))
        s_vals = 0.5 + sigma + 1j * t_grid
        J_vals = np.array([J(complex(s), primes, local_outer) for s in s_vals])
        theta_vals = (2.0 * J_vals - 1.0) / (2.0 * J_vals + 1.0)
        # Two equivalent Pick kernels: direct and via Herglotz factorization
        K_direct = pick_matrix(theta_vals, s_vals)
        K_fact = pick_matrix_from_J(J_vals, s_vals)
        # Symmetrize and compare numerically
        K = 0.5 * (K_direct + K_fact)
        w = np.linalg.eigvalsh(K)
        minima_pick.append(float(w.min().real))
        # Also track Herglotz kernel PSD witness
        H = herglotz_matrix(J_vals, s_vals)
        wh = np.linalg.eigvalsh(H)
        herglotz_minima.append(float(wh.min().real))
        if w.min() < -tol:
            return False, minima_pick, herglotz_minima
    return True, minima_pick, herglotz_minima


def certify_cover(
    intervals: Sequence[Tuple[float, float]],
    Pmax: int = 100000,
    outer_global_range: Optional[Tuple[float, float]] = None,
    **kwargs,
) -> List[dict]:
    primes = primes_upto(Pmax)
    sigmas = kwargs.get('sigmas', (2e-2, 1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4))
    eps_factors = kwargs.get('eps_factors', (1.0, 0.5, 0.25))
    n_outer = kwargs.get('n_outer', 801)
    outer_cache: Optional[Dict[float, Outer]] = None
    if kwargs.get('build_outer') and outer_global_range is not None:
        Tlo, Thi = outer_global_range
        outer_cache = build_outer_global(Tlo, Thi, sigmas, primes, eps_factors, n_outer=n_outer)
    # filter unsupported kwargs for certify_interval
    fwd = dict(kwargs)
    fwd.pop('n_outer', None)
    fwd.pop('outer_global_range', None)
    results = []
    for (T1, T2) in intervals:
        ok, min_pick, min_herg = certify_interval(T1, T2, primes, outer_per_sigma=outer_cache, **fwd)
        results.append({"interval": (T1, T2), "certified": ok, "min_eigs_pick": min_pick, "min_eigs_herglotz": min_herg})
        print(f"I=[{T1},{T2}] -> {'OK' if ok else 'FAIL'}; min eigs (Pick): {min_pick}; min eigs (Herglotz): {min_herg}")
    return results


def certify_cover_logged(
    intervals: Sequence[Tuple[float, float]],
    outfile: str,
    Pmax: int = 100000,
    outer_global_range: Optional[Tuple[float, float]] = None,
    **kwargs,
) -> None:
    primes = primes_upto(Pmax)
    sigmas = kwargs.get('sigmas', (2e-2, 1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4))
    eps_factors = kwargs.get('eps_factors', (1.0, 0.5, 0.25))
    n_outer = kwargs.get('n_outer', 801)
    outer_cache: Optional[Dict[float, Outer]] = None
    if kwargs.get('build_outer') and outer_global_range is not None:
        Tlo, Thi = outer_global_range
        outer_cache = build_outer_global(Tlo, Thi, sigmas, primes, eps_factors, n_outer=n_outer)
    meta = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "Pmax": Pmax,
        "kwargs": {k: v for k, v in kwargs.items() if k != "outer"},
    }
    with open(outfile, "a") as f:
        f.write(json.dumps({"meta": meta}) + "\n")
        # filter unsupported kwargs for certify_interval
        fwd = dict(kwargs)
        fwd.pop('n_outer', None)
        fwd.pop('outer_global_range', None)
        for (T1, T2) in intervals:
            ok, min_pick, min_herg = certify_interval(T1, T2, primes, outer_per_sigma=outer_cache, **fwd)
            rec = {
                "interval": [T1, T2],
                "certified": ok,
                "min_eigs_pick": min_pick,
                "min_eigs_herglotz": min_herg,
            }
            f.write(json.dumps(rec) + "\n")
            print(f"I=[{T1},{T2}] -> {'OK' if ok else 'FAIL'}; min eigs (Pick): {min_pick}; min eigs (Herglotz): {min_herg}")


# ------------------------------
# CLI
# ------------------------------

def parse_list_floats(s: str) -> List[float]:
    return [float(x) for x in s.split(',') if x]


def main():
    ap = argparse.ArgumentParser(description="Pick-matrix (P+) certification driver")
    ap.add_argument('--range', nargs=2, type=float, metavar=('TLO','THI'), help='cover range [TLO,THI]')
    ap.add_argument('--step', type=float, default=10.0, help='interval length for cover')
    ap.add_argument('--intervals', type=str, help='semicolon-separated list of a,b intervals')
    ap.add_argument('--pmax', type=int, default=50000, help='prime cutoff')
    ap.add_argument('--build-outer', action='store_true', help='build uniform-epsilon outer per interval/sigma')
    ap.add_argument('--outer-global', nargs=2, type=float, metavar=('TLO','THI'), help='build one outer per sigma on [TLO,THI] and reuse for all intervals')
    ap.add_argument('--outer-module', type=str, help='python module path providing an outer function O(s)')
    ap.add_argument('--outer-func', type=str, default='O', help='function name in outer-module (default: O)')
    ap.add_argument('--sigmas', type=str, help='comma list of sigmas')
    ap.add_argument('--ns', type=str, help='comma list of grid sizes')
    ap.add_argument('--eps-factors', type=str, help='comma list of epsilon factors for outer averaging')
    ap.add_argument('--n-outer', type=int, default=401, help='outer grid size (odd preferred)')
    ap.add_argument('--log', type=str, help='write JSONL log to this file')
    ap.add_argument('--dps', type=int, default=50, help='mpmath precision (decimal digits)')
    ap.add_argument('--quick', action='store_true', help='use a lighter preset (faster)')
    ap.add_argument('--scan-phase', action='store_true', help='run phase-cone scan instead of Pick-PSD')
    ap.add_argument('--phase-sigma', type=float, default=1e-3, help='sigma for phase-cone scan')
    ap.add_argument('--phase-n', type=int, default=4096, help='grid size for phase-cone scan')
    ap.add_argument('--phase-pad', type=float, default=20.0, help='padding (t-units) for phase-cone scan')
    ap.add_argument('--det2', type=str, default='series', choices=['series','product'], help='choose det2 model (series or product)')
    ap.add_argument('--kmax', type=int, default=12, help='kmax for series det2')
    args = ap.parse_args()

    global DET2_FUNC, DET2_KMAX
    if args.det2 == 'series':
        DET2_KMAX = args.kmax
        DET2_FUNC = lambda s, primes: det2_I_minus_A_series(s, primes, DET2_KMAX)
    else:
        DET2_FUNC = det2_I_minus_A_product

    mp.mp.dps = args.dps

    intervals: List[Tuple[float,float]] = []
    if args.intervals:
        parts = args.intervals.split(';')
        for p in parts:
            a, b = p.split(',')
            intervals.append((float(a), float(b)))
    elif args.range:
        Tlo, Thi = args.range
        cur = Tlo
        while cur < Thi - 1e-12:
            intervals.append((cur, min(cur + args.step, Thi)))
            cur += args.step
    else:
        intervals = [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0), (30.0, 40.0)]

    # Presets
    sigmas = None
    ns = None
    eps_factors = None
    if args.quick:
        sigmas = [1e-2, 5e-3, 1e-3]
        ns = [64, 128, 192]
        eps_factors = [1.0, 0.5]
    if args.sigmas:
        sigmas = parse_list_floats(args.sigmas)
    if args.ns:
        ns = [int(x) for x in args.ns.split(',') if x]
    if args.eps_factors:
        eps_factors = parse_list_floats(args.eps_factors)

    # Optional plugin outer
    outer_plugin: Optional[Outer] = None
    if args.outer_module:
        outer_plugin = load_outer_from_module(args.outer_module, args.outer_func)

    if args.scan_phase:
        primes = primes_upto(args.pmax)
        scan_phase_cover(intervals, sigma=args.phase_sigma, primes=primes, outer=outer_plugin, n_phase=args.phase_n, pad=args.phase_pad, outfile=args.log)
        return

    kwargs = {}
    if sigmas is not None:
        kwargs['sigmas'] = sigmas
    if ns is not None:
        kwargs['ns'] = ns
    if eps_factors is not None:
        kwargs['eps_factors'] = eps_factors
    if args.build_outer:
        kwargs['build_outer'] = True
    if args.n_outer:
        kwargs['n_outer'] = args.n_outer
    if outer_plugin is not None:
        kwargs['outer'] = outer_plugin

    if args.log:
        certify_cover_logged(intervals, outfile=args.log, Pmax=args.pmax, outer_global_range=tuple(args.outer_global) if args.outer_global else None, **kwargs)
    else:
        certify_cover(intervals, Pmax=args.pmax, outer_global_range=tuple(args.outer_global) if args.outer_global else None, **kwargs)


if __name__ == "__main__":
    main()


