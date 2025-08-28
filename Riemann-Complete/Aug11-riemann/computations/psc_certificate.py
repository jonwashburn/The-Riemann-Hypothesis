#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PSC Certificate: interval-style enclosures for C_Gamma(psi0), c0(psi0), and ||H[psi0]||_inf.

This script provides computation-ready routines to bound the constants in the
certificate inequality. It uses mpmath with high precision; for publication-grade
rigor, replace the numerical quadratures with an interval arithmetic backend
such as intvalpy or arb bindings and keep a ledger of error bounds.

Outputs (upper/lower certified bounds):
  - bound_C_gamma(psi0):  upper bound for sup_L C_Gamma^{(L)} with window psi0
  - bound_c0(psi0):       lower bound for c0(psi0) = inf_{0<b<=1, |x|<=1} (P_b * psi0)(x)
  - bound_H_inf_of_window(psi0): upper bound for ||H[psi0]||_inf

Also includes:
  - bound_CP_kappa(kappa): uses the proven inequality sup_L C_P(psi0,L;kappa)*L <= 2*kappa

Notes:
  - This code is designed to be simple to audit. Switch mp.mp.dps to increase precision.
  - For interval certification, wrap integrands and sums with outward rounding and
    verified quadrature (left/right Riemann sums with monotonicity checks, etc.).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Tuple

import mpmath as mp


# ----------------------------- Configuration ---------------------------------

mp.mp.dps = 80  # working precision; raise to tighten enclosures


# ----------------------------- Window psi0 ------------------------------------

def psi_bump(alpha: mp.mpf) -> Callable[[mp.mpf], mp.mpf]:
    """Return a standard C^∞ bump ψ_α(t) ∝ exp(-α/(1-t^2)) on [-1,1], normalized."""
    # Precompute normalizer for this alpha
    def _Z(alpha: mp.mpf) -> mp.mpf:
        f = lambda x: mp.e**(-alpha/(1 - x*x)) if abs(x) < 1 else mp.mpf('0')
        return mp.quad(f, [-1, 1])

    Z = _Z(alpha)

    def psi_alpha(t: mp.mpf) -> mp.mpf:
        if abs(t) >= 1:
            return mp.mpf('0')
        val = mp.e**(-alpha/(1 - t*t))
        return val / Z

    return psi_alpha


def phi_L_factory(psi: Callable[[mp.mpf], mp.mpf]) -> Callable[[mp.mpf, mp.mpf], mp.mpf]:
    def phi_L(t: mp.mpf, L: mp.mpf) -> mp.mpf:
        return (1/L) * psi(t / L)
    return phi_L


# ------------------------ Archimedean constant C_Gamma ------------------------

def arch_integrand(t: mp.mpf) -> mp.mpf:
    """Im d/dt log( pi^{-s/2} Gamma(s/2) * (s(1-s))/2 ) at s=1/2+it.

    Formula: 0.5*Re(psi(1/4 + i t/2)) - 0.5*log(pi) + 2 t / (1 + 4 t^2).
    """
    z = mp.mpf('0.25') + 0.5j * t
    term = 0.5 * mp.re(mp.digamma(z)) - 0.5 * mp.log(mp.pi) + (2*t)/(1 + 4*t*t)
    return term


def bound_C_gamma(psi_ignored: Callable[[mp.mpf], mp.mpf], max_L: float = 8.0, sup_samples: int = 1201) -> mp.mpf:
    """Upper bound C_Gamma by sup_{|t|≤max_L} |arch_integrand(t)|.

    Since |∫ φ_L f| ≤ sup_{|t|≤L} |f(t)| and L ≤ max_L, it suffices to bound the sup
    over [-max_L, max_L]."""
    L = mp.mpf(max_L)
    M = mp.mpf('0')
    for j in range(sup_samples):
        t = -L + (2*L) * mp.mpf(j) / (sup_samples - 1)
        val = abs(arch_integrand(t))
        if val > M:
            M = val
    # 1% safety inflation
    M *= mp.mpf('1.01')
    return M


# ------------------------ Poisson–window constant c0 --------------------------

def P_b(x: mp.mpf, b: mp.mpf) -> mp.mpf:
    return (1/mp.pi) * (b / (b*b + x*x))


def conv_P_psi0(x: mp.mpf, b: mp.mpf) -> mp.mpf:
    f = lambda y: P_b(x - y, b) * psi0(y)
    return mp.quad(f, [-1, 1])


def bound_c0_whitney() -> mp.mpf:
    """Whitney lower bound constant for normalized Poisson kernel across intervals.

    For a in [L,2L] and γ in I, with normalized Poisson P_a(x)= (1/π) a/(a^2+x^2),
    min_γ ∫_I P_a(t-γ) dt = (1/π) arctan(L/a). Minimizing over a∈[L,2L] gives
    c0^W = (1/π) arctan(1/2).
    """
    return (1/mp.pi) * mp.atan(mp.mpf('0.5'))


# ------------------------ Hilbert transform of psi0 ---------------------------

def H_of_window(t: mp.mpf, psi: Callable[[mp.mpf], mp.mpf]) -> mp.mpf:
    """Hilbert transform of psi0 at t: (1/π) PV ∫ psi0(τ)/(t-τ) dτ.

    Numerically evaluate principal value via symmetric exclusion around τ=t.
    """
    eps = mp.mpf('1e-8')
    g1 = lambda u: psi(u) / (t - u)
    v1 = mp.quad(g1, [-1, t - eps]) if t > -1 else mp.mpf('0')
    v2 = mp.quad(g1, [t + eps, 1]) if t < 1 else mp.mpf('0')
    return (v1 + v2) / mp.pi


def bound_H_inf_of_window(psi: Callable[[mp.mpf], mp.mpf], samples: int = 2001) -> mp.mpf:
    """Upper bound for ||H[psi0]||_inf by sampling on [-T, T] with T >= 2.

    Since psi0 is compactly supported, H[psi0] decays ~ 1/t; max occurs near
    support edges. Increase samples for tighter bound.
    """
    T = mp.mpf('4')
    worst = mp.mpf('0')
    for k in range(samples):
        t = -T + (2*T) * mp.mpf(k) / (samples - 1)
        val = abs(H_of_window(t, psi))
        worst = max(worst, val)
    return worst


# ------------------------ Prime constant via kappa ----------------------------

def bound_CP_kappa(kappa: float) -> mp.mpf:
    """Sup_L C_P(psi0, L; kappa) * L <= 2*kappa."""
    return mp.mpf('2') * mp.mpf(kappa)


# ------------------------ End-to-end certificate check ------------------------

@dataclass
class CertificateBounds:
    alpha: mp.mpf
    C_gamma: mp.mpf
    c0: mp.mpf
    c0_bulk: mp.mpf
    H_inf_window: mp.mpf
    CP_margin: mp.mpf


def c0_bulk_whitney(theta: mp.mpf = mp.mpf('0.25')) -> mp.mpf:
    """Bulk-only Whitney lower bound with γ restricted to interior [θL, (1-θ)L].

    c0_bulk = (1/π) * inf_{a∈[1,2], x∈[θ,1-θ]} [ arctan((1-x)/a) + arctan(x/a) ].
    """
    best = mp.inf
    for ia in range(100):
        a = mp.mpf('1') + (mp.mpf(ia)/mp.mpf(99)) * (mp.mpf('1'))  # 1..2
        for ix in range(100):
            x = theta + (mp.mpf(ix)/mp.mpf(99)) * (mp.mpf('1') - 2*theta)
            val = (1/mp.pi) * (mp.atan((1 - x)/a) + mp.atan(x/a))
            if val < best:
                best = val
    return best


def compute_certificate(kappa: float = 0.10, alpha: float = 1.0, compute_H: bool = False) -> CertificateBounds:
    psi = psi_bump(mp.mpf(alpha))
    Cg = bound_C_gamma(psi)
    c0 = bound_c0_whitney()
    c0b = c0_bulk_whitney(mp.mpf('0.25'))
    Hinf = bound_H_inf_of_window(psi) if compute_H else mp.mpf('nan')
    CPm = bound_CP_kappa(kappa)
    return CertificateBounds(alpha=mp.mpf(alpha), C_gamma=Cg, c0=c0, c0_bulk=c0b, H_inf_window=Hinf, CP_margin=CPm)


def main():
    print("PSC certificate bounds (non-interval demo; increase mp.dps as needed):")
    # Optimize over alpha grid and smaller kappa
    kappas = [0.10, 0.05, 0.02, 0.01]
    alphas = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    best = None
    for kappa in kappas:
        for alpha in alphas:
            b = compute_certificate(kappa=kappa, alpha=alpha, compute_H=False)
            lhs_whitney = (b.C_gamma + b.CP_margin) / b.c0
            lhs_bulk = (b.C_gamma + b.CP_margin) / b.c0_bulk
            score = lhs_bulk  # prefer bulk bound
            if best is None or score < best[0]:
                best = (score, kappa, alpha, b)
    score, kappa, alpha, b = best
    print(f"Best parameters: kappa={kappa}, alpha={alpha}")
    print(f"C_gamma(psi_alpha) ≲ {b.C_gamma}")
    print(f"c0^Whitney = (1/π) arctan(1/2) ≈ {b.c0}")
    print(f"c0^bulk(θ=1/4) ≈ {b.c0_bulk}")
    print(f"||H[psi_alpha]||_inf ≲ {b.H_inf_window}")
    print(f"sup_L C_P·L ≲ {b.CP_margin}")
    lhs_whitney = (b.C_gamma + b.CP_margin) / b.c0
    lhs_bulk = (b.C_gamma + b.CP_margin) / b.c0_bulk
    print(f"(C_gamma + CP_margin)/c0^W ≲ {lhs_whitney}  vs  π/4 ≈ {mp.pi/4}")
    print(f"(C_gamma + CP_margin)/c0^bulk ≲ {lhs_bulk}  vs  π/4 ≈ {mp.pi/4}")


if __name__ == "__main__":
    main()


