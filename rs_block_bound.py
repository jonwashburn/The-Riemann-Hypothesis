#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Finite block recipe helper for D_ε on K_σ.

Implements the user's exact formulas (with the standard sign correction in (⋆)):

  (⋆)  Choose the smallest integer P such that the prime tail
       ∑_{p>P} p^{-2σ} ≤ (ε/2)^2, using a selected explicit bound method
       (default: Rosser–Schoenfeld (P1) with x≥17).

  (4)  For each prime p ≤ P,
       N_p := max(1, ceil( ( log(4 π(P) / ε^2) - log(1 - p^{-2σ}) ) / (2 σ log p) - 1 ) ).

  (B)  B(σ; P, {N_p}) := sqrt( TailBound(σ,P) + sum_{p ≤ P} p^{-2σ(N_p+1)} / (1 - p^{-2σ}) ),
       where TailBound(σ,P) is computed by the same chosen explicit method as in (⋆).

Notes
-----
- The sign in the denominator of the prime-tail bound must be (2σ - 1) > 0 (standard
  Rosser–Schoenfeld-style tail), not (1 - 2σ).
- T does not enter the certification formulas, but is accepted as an input for interface parity.
"""

from __future__ import annotations

import argparse
import math
from typing import List, Literal


def sieve_primes_upto(n: int) -> List[int]:
    """Return all primes ≤ n via a simple sieve. Fast enough for up to ~1e7.
    For larger n, consider a segmented sieve. """
    if n < 2:
        return []
    sieve = bytearray(b"\x01") * (n + 1)
    sieve[0:2] = b"\x00\x00"
    limit = int(n ** 0.5) + 1
    for p in range(2, limit):
        if sieve[p]:
            step = p
            start = p * p
            sieve[start:n + 1:step] = b"\x00" * (((n - start) // step) + 1)
    return [i for i in range(n + 1) if sieve[i]]


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    r = int(n ** 0.5)
    f = 3
    while f <= r:
        if n % f == 0:
            return False
        f += 2
    return True


def next_prime(n: int) -> int:
    k = max(2, n + 1)
    while not is_prime(k):
        k += 1
    return k


def sum_primes_upto(n: int, sigma: float) -> float:
    if n < 2:
        return 0.0
    ps = sieve_primes_upto(n)
    return sum((p ** (-sigma)) for p in ps)


K_ROSSER = 1.25506  # Rosser–Schoenfeld constant for π(x) ≤ K x / log x, x ≥ 17


def _U_p1(alpha: float, P: int, K: float = K_ROSSER) -> float:
    """(P1) Rosser–Schoenfeld tail bound for ∑_{p>P} p^{-α}, valid for P ≥ 17, α>1:
       U(P) = (K * α / (α - 1)) * P^{1-α} / log P.
    """
    if alpha <= 1:
        raise ValueError("alpha must be > 1.")
    if P < 17:
        raise ValueError("P must be ≥ 17 for (P1).")
    return (K * alpha / (alpha - 1.0)) * (P ** (1.0 - alpha)) / math.log(P)


def _U_dusart(alpha: float, P: int) -> float:
    """Dusart-style tail: using π(x) ≤ x / (log x - 1) for x ≥ 599.
       Gives U(P) = α / ((α - 1)(log P - 1)) * P^{1-α}.
    """
    if alpha <= 1:
        raise ValueError("alpha must be > 1.")
    if P < 599:
        raise ValueError("P must be ≥ 599 for Dusart bound.")
    return alpha / ((alpha - 1.0) * (math.log(P) - 1.0)) * (P ** (1.0 - alpha))


def _U_triv(alpha: float, P: int) -> float:
    """Trivial integer tail: ∑_{p>P} p^{-α} ≤ ∑_{n>P} n^{-α} ≤ P^{1-α} / (α - 1)."""
    if alpha <= 1:
        raise ValueError("alpha must be > 1.")
    return (P ** (1.0 - alpha)) / (alpha - 1.0)


def minimal_P_for_tail(alpha: float, eta: float, bound: Literal["p1", "dusart", "triv"] = "p1") -> int:
    """Return minimal integer P satisfying TailBound(P) ≤ eta, using selected method.
       Methods:
         - p1: Rosser–Schoenfeld (P1), P ≥ 17
         - dusart: π(x) ≤ x/(log x - 1), P ≥ 599
         - triv: integer tail, P ≥ 2
    """
    if alpha <= 1:
        raise ValueError("alpha must be > 1.")
    if eta <= 0.0:
        raise ValueError("eta must be positive.")

    if bound == "p1":
        P_min = 17
        U = lambda P: _U_p1(alpha, P)
    elif bound == "dusart":
        P_min = 599
        U = lambda P: _U_dusart(alpha, P)
    elif bound == "triv":
        P_min = 2
        U = lambda P: _U_triv(alpha, P)
    else:
        raise ValueError("Unknown bound method.")

    # If already satisfied at the minimum
    if U(P_min) <= eta:
        return P_min

    # Safe upper bracket via trivial tail
    beta = alpha - 1.0
    P_hi_guess = int(math.ceil((1.0 / (beta * eta)) ** (1.0 / beta)))
    P_hi = max(P_min, P_hi_guess)

    # Binary search on [P_min, P_hi]
    lo, hi = P_min, P_hi
    while lo < hi:
        mid = (lo + hi) // 2
        if U(mid) <= eta:
            hi = mid
        else:
            lo = mid + 1
    return lo


def choose_P(sigma: float, eps: float, bound: Literal["p1", "dusart", "triv"] = "p1") -> int:
    """Choose minimal P s.t. ∑_{p>P} p^{-2σ} ≤ (ε/2)^2 by the selected method."""
    if sigma <= 0.5:
        raise ValueError("sigma must be > 1/2.")
    alpha = 2.0 * sigma
    eta = (eps / 2.0) ** 2
    return minimal_P_for_tail(alpha, eta, bound=bound)


def compute_N_p(p: int, sigma: float, eps: float, pi_P: int) -> int:
    """N_p per formula (4):
       N_p := max(1, ceil( ( log(4 π(P) / ε^2) - log(1 - p^{-2σ}) ) / (2 σ log p) - 1 ) ).
    """
    two_sigma = 2.0 * sigma
    if two_sigma <= 1.0:
        raise ValueError("sigma must satisfy 2*sigma > 1.")
    if p < 2:
        return 1
    # Guard: 1 - p^{-2σ} > 0
    one_minus = 1.0 - p ** (-two_sigma)
    if one_minus <= 0.0:
        # Should not happen for p≥2, σ>1/2
        one_minus = 1e-300
    numerator = math.log(4.0 * max(1, pi_P) / (eps ** 2)) - math.log(one_minus)
    denom = two_sigma * math.log(p)
    # If denom somehow underflows, fall back to 1
    if denom <= 0.0:
        return 1
    raw = (numerator / denom) - 1.0
    Np = int(math.ceil(raw))
    return max(1, Np)


def compute_B(sigma: float, P: int, primes: List[int], Np_list: List[int], bound: Literal["p1", "dusart", "triv"] = "p1") -> float:
    """Compute B per (B):
       B^2 = TailBound(σ,P) + sum_{p≤P} p^{-2σ(N_p+1)} / (1 - p^{-2σ}).
    """
    two_sigma = 2.0 * sigma
    alpha = two_sigma
    if bound == "p1":
        tail = _U_p1(alpha, max(P, 17))
    elif bound == "dusart":
        tail = _U_dusart(alpha, max(P, 599))
    elif bound == "triv":
        tail = _U_triv(alpha, max(P, 2))
    else:
        raise ValueError("Unknown bound method.")
    s = 0.0
    for p, Np in zip(primes, Np_list):
        denom = 1.0 - p ** (-two_sigma)
        if denom <= 0.0:
            # Should not happen; safeguard
            continue
        term = p ** (-two_sigma * (Np + 1)) / denom
        s += term
    return math.sqrt(tail + s)


def main():
    ap = argparse.ArgumentParser(description="Compute P (⋆), N_p (4), and B (B) for given (σ, T, ε).")
    ap.add_argument("--sigma", type=float, required=True, help="Real part lower bound σ (e.g., 0.6)")
    ap.add_argument("--T", type=float, required=False, default=50.0, help="Imag height bound (not used in formulas)")
    ap.add_argument("--eps", type=float, required=True, help="Small parameter ε (e.g., 0.10)")
    ap.add_argument("--bound", type=str, choices=["p1", "dusart", "triv"], default="p1", help="Tail bound method: p1 (Rosser–Schoenfeld), dusart, triv")
    # Finite-block spectral gap parameters
    ap.add_argument("--block_model", type=str, choices=["unweighted", "weighted", "weighted_padaptive"], default="unweighted", help="Block model for off-diagonal norms: unweighted (√(N_p N_q)(pq)^{-σ}), weighted (¼·(pq)^{-σ}), or weighted_padaptive (¼·p^{-(σ+1/2)}q^{-(σ+1/2)})")
    ap.add_argument("--mu_min", type=float, default=0.05, help="Uniform in-block Gershgorin lower bound μ_min (post-normalization)")
    ap.add_argument("--Cwin", type=float, default=1.0, help="Window/basis constant Cwin in Upq bounds")
    # Small-prime disentangling (P3)
    ap.add_argument("--small_Q", type=int, default=0, help="Excise primes ≤ Q for P3 disentangling (0 disables)")
    ap.add_argument("--Nmax", type=int, default=0, help="N_max for unweighted P3 lift (ignored for weighted)")
    ap.add_argument("--Q", type=int, default=0, help="If >0, compute S_{σ0}(Q) exactly over primes ≤ Q and report Δ_w / Δ_unwt")
    ap.add_argument("--mu_min_auto", action="store_true", help="If set, set μ_min = 1 - L(p_min)/6 with p_min=nextprime(Q) for weighted models; else use input μ_min")
    ap.add_argument("--emit_tex", type=str, default="", help="If set, write a small LaTeX certificate block to this path")
    ap.add_argument("--cover_only", action="store_true", help="If set, skip heavy (⋆)/(4)/(B) computations and only emit covering outputs")
    # Covering / per-σ budgets and margins
    ap.add_argument("--cov_sigma_start", type=float, default=0.0, help="Start σ for covering grid (0 disables covering)")
    ap.add_argument("--cov_sigma_end", type=float, default=0.0, help="End σ for covering grid (ignored if start=0)")
    ap.add_argument("--theta_max", type=float, default=0.5, help="Max per-step θ = K(σ)·h (e.g., 0.5 or 0.45)")
    ap.add_argument("--h_max", type=float, default=0.02, help="Max step size in σ")
    ap.add_argument("--pmin", type=int, default=0, help="If >0, use this far cutoff prime p_min; else use next_prime(Q)")
    ap.add_argument("--pmin_max", type=int, default=200000, help="Max p_min to try when adapting per-σ")
    ap.add_argument("--margin_min", type=float, default=0.0, help="Target per-row margin: δ_cert(σ) - δ0·e^{-L(σ)} ≥ margin_min")
    ap.add_argument("--emit_cover_csv", type=str, default="", help="If set, write per-σ covering CSV to this path")
    ap.add_argument("--emit_cover_tex", type=str, default="", help="If set, write per-σ covering LaTeX table to this path")
    args = ap.parse_args()

    sigma = args.sigma
    T = args.T
    eps = args.eps

    if not args.cover_only:
        # Choose P via selected bound
        P = choose_P(sigma, eps, bound=args.bound)

        # Sieve primes ≤ P and compute π(P)
        primes = sieve_primes_upto(P)
        piP = len(primes)

        # Compute N_p per (4)
        Np_list = [compute_N_p(p, sigma, eps, piP) for p in primes]

        # Compute B per (B)
        B = compute_B(sigma, P, primes, Np_list, bound=args.bound)

        # Report
        print("Inputs:")
        print(f"  sigma = {sigma}")
        print(f"  T     = {T}  (not used in formulas)")
        print(f"  eps   = {eps}")
        print()
        print("Construction:")
        print(f"  P (from (⋆), {args.bound}) = {P}")
        print(f"  #primes ≤ P        = {piP}")
        if primes:
            preview = min(5, len(primes))
            print("  First primes and N_p (p, N_p):")
            print("   ", [(primes[i], Np_list[i]) for i in range(preview)])
        print()
        print("Bound (B):")
        print(f"  B(σ;P,{{N_p}})     = {B}")
        print(f"  Target ε           = {eps}")
        print(f"  Certification      = {'OK' if B <= eps else 'FAILS (B>ε)'}")

    # ===== Finite-block spectral gap δ(σ0) using model bounds =====
    if not args.cover_only:
        two_sigma = 2.0 * sigma
        # Precompute q^{-σ} and √Nq q^{-σ}
        q_pow = [p ** (-sigma) for p in primes]
        sqrtN = [math.sqrt(max(n, 1)) for n in Np_list]
        sqrtN_qpow = [sqrtN[i] * q_pow[i] for i in range(len(primes))]

        def sum_offdiag_for_p(idx_p: int) -> float:
            p = primes[idx_p]
            if args.block_model == "unweighted":
                # Upq ≤ Cwin √(N_p N_q) (pq)^{-σ}
                factor_p = args.Cwin * math.sqrt(max(Np_list[idx_p], 1)) * (p ** (-sigma))
                s = 0.0
                for j in range(len(primes)):
                    if j == idx_p:
                        continue
                    s += sqrtN_qpow[j]
                return factor_p * s
            elif args.block_model == "weighted":
                # weighted: Upq ≤ (Cwin/4) (pq)^{-σ}
                factor_p = (args.Cwin / 4.0) * (p ** (-sigma))
                s = 0.0
                for j in range(len(primes)):
                    if j == idx_p:
                        continue
                    s += q_pow[j]
                return factor_p * s
            else:
                # weighted_padaptive: Upq ≤ (Cwin/4) p^{-(σ+1/2)} q^{-(σ+1/2)}
                factor_p = (args.Cwin / 4.0) * (p ** (-(sigma + 0.5)))
                s = 0.0
                for j in range(len(primes)):
                    if j == idx_p:
                        continue
                    s += (primes[j] ** (-(sigma + 0.5)))
                return factor_p * s

        offdiag_sums = [sum_offdiag_for_p(i) for i in range(len(primes))]
        if offdiag_sums:
            max_off = max(offdiag_sums)
        else:
            max_off = 0.0

        # μ_min selection: if weighted model and mu_min_auto with Q>0, set μ_min = 1 - L(p_min)/6
        if args.mu_min_auto and args.block_model in ("weighted", "weighted_padaptive") and args.Q and args.Q > 0:
            pmin = next_prime(args.Q)
            L_pmin = (1.0 - sigma) * math.log(pmin) * (pmin ** (-sigma))
            mu_min = 1.0 - L_pmin / 6.0
        else:
            mu_min = max(args.mu_min, 0.0)
        delta_BG = mu_min - max_off
        # With uniform μ, Schur–Weyl reduces to same max off-diagonal sum
        delta_SW = mu_min - max_off
        delta = max(0.0, delta_BG, delta_SW)

        print()
        print("Finite-block spectral gap (model certificate):")
        print(f"  block_model        = {args.block_model}")
        print(f"  μ_min (input)      = {mu_min}")
        print(f"  Cwin               = {args.Cwin}")
        print(f"  max Σ_q≠p U_pq     = {max_off}")
        print(f"  δ_BG               = {delta_BG}")
        print(f"  δ_SW               = {delta_SW}")
        print(f"  δ(σ0)              = {delta}")

    # Optionally emit a LaTeX snippet with the main figures
    if args.emit_tex and not args.cover_only:
        try:
            def median(lst: List[int]) -> float:
                n = len(lst)
                if n == 0:
                    return 0.0
                srt = sorted(lst)
                mid = n // 2
                if n % 2 == 1:
                    return float(srt[mid])
                return 0.5 * (srt[mid - 1] + srt[mid])

            N_min = min(Np_list) if Np_list else 0
            N_med = median(Np_list)
            N_max = max(Np_list) if Np_list else 0
            with open(args.emit_tex, "w", encoding="utf-8") as f:
                f.write("% Auto-generated certificate snippet\n")
                f.write("%% Parameters: sigma=%.6g, eps=%.6g, bound=%s\n" % (sigma, eps, args.bound))
                f.write("%% P=%d, pi(P)=%d, N_p(min/med/max)=(%d, %.1f, %d)\n" % (P, piP, N_min, N_med, N_max))
                f.write("%% B=%.6g, delta=%.6g, model=%s, mu_min=%.6g, Cwin=%.6g\n" % (B, delta, args.block_model, mu_min, args.Cwin))
                f.write("\\begin{itemize}\n")
                f.write("  \\item \\textbf{Half-plane:} $\\sigma_0=%.3f$\n" % sigma)
                f.write("  \\item \\textbf{Prime cutoff:} $P=%d$, $\\pi(P)=%d$\n" % (P, piP))
                f.write("  \\item \\textbf{Depths:} $N_p^{\\min}=%d$, $N_p^{\\mathrm{med}}=%.1f$, $N_p^{\\max}=%d$\n" % (N_min, N_med, N_max))
                f.write("  \\item \\textbf{Tail/certificate:} using %s, $B=%.4f$ (target $\\varepsilon=%.3f$)\n" % (args.bound, B, eps))
                safe_model = args.block_model.replace("_", "\\_")
                f.write("  \\item \\textbf{Finite-block gap (model):} $\\delta(\\sigma_0)=%.4f$ (%s, $\\mu_{\\min}=%.3f$, $C_{\\mathrm{win}}=%.3f$)\n" % (delta, safe_model, mu_min, args.Cwin))
                if args.Q and args.Q > 0:
                    pmin = next_prime(args.Q)
                    S_sigQ = sum_primes_upto(args.Q, sigma)
                    L_pmin = (1.0 - sigma) * math.log(pmin) * (pmin ** (-sigma))
                    muL_min = 1.0 - L_pmin / 6.0
                    Delta_w = 0.25 * (pmin ** (-sigma)) * S_sigQ
                    f.write("  \\item Small-prime sum: $S_{\\sigma_0}(%d)=%.6f$, $p_{\\min}=%d$\n" % (args.Q, S_sigQ, pmin))
                    f.write("  \\item Far-block in-block bound: $L(p_{\\min})=%.6f$, $\\mu^{\\mathrm L}_{\\min}=1-\\tfrac{1}{6}L(p_{\\min})=%.6f$\n" % (L_pmin, muL_min))
                    if args.block_model == "unweighted":
                        f.write("  \\item Uniform far\(\\rightarrow\)small budget: $\\Delta_{\\rm unwt}=N_{\\max}\,p_{\\min}^{-\\sigma_0}\,S_{\\sigma_0}(%d)$ (set $N_{\\max}=%d$)\n" % (args.Q, args.Nmax))
                    else:
                        f.write("  \\item Uniform far\(\\rightarrow\)small budget: $\\Delta_w=\\tfrac14\,p_{\\min}^{-\\sigma_0}\,S_{\\sigma_0}(%d)=%.6f$\n" % (args.Q, Delta_w))
                    # If weighted p-adaptive, also print sigma* budgets and δ_cert
                    if args.block_model == "weighted_padaptive":
                        sigma_star = sigma + 0.5
                        S_sigstar_Q = sum_primes_upto(args.Q, sigma_star)
                        # Integer tail majorant from pmin - 1
                        if sigma_star <= 1.0:
                            tail_majorant = float('inf')
                        else:
                            tail_majorant = ((pmin - 1) ** (1.0 - sigma_star)) / (sigma_star - 1.0)
                        # Δ budgets
                        C4 = args.Cwin / 4.0
                        pmin_sigstar = pmin ** (-sigma_star)
                        Delta_FS = C4 * pmin_sigstar * S_sigstar_Q
                        Delta_FF = C4 * pmin_sigstar * tail_majorant
                        # Small-block worst μ via max L(p) over p≤Q
                        small_primes = sieve_primes_upto(args.Q)
                        L_values_small = [(p, (1.0 - sigma) * math.log(p) * (p ** (-sigma))) for p in small_primes if p <= args.Q]
                        L_max_small = max(L for p, L in L_values_small) if L_values_small else 0.0
                        mu_small_min = 1.0 - L_max_small / 6.0
                        # Δ_SS and Δ_SF with worst small-row p=2 (if 2≤Q)
                        two_term = (2 ** (-sigma_star)) if args.Q >= 2 else 0.0
                        sum_small_excl2 = max(S_sigstar_Q - (2 ** (-sigma_star)), 0.0) if args.Q >= 2 else S_sigstar_Q
                        Delta_SS = C4 * two_term * sum_small_excl2
                        Delta_SF = C4 * two_term * tail_majorant
                        # δ_cert
                        delta_small_rows = mu_small_min - (Delta_SS + Delta_SF)
                        delta_far_rows = muL_min - (Delta_FS + Delta_FF)
                        delta_cert = min(delta_small_rows, delta_far_rows)
                        f.write("  \\item $S_{\\sigma_0+1/2}(%d)=%.6f$, integer tail$=%.6f$\n" % (args.Q, S_sigstar_Q, tail_majorant))
                        f.write("  \\item Budgets: $\\Delta_{\\mathrm{FS}}=%.6f$, $\\Delta_{\\mathrm{FF}}=%.6f$, $\\Delta_{\\mathrm{SS}}=%.6f$, $\\Delta_{\\mathrm{SF}}=%.6f$\n" % (Delta_FS, Delta_FF, Delta_SS, Delta_SF))
                        f.write("  \\item $\\mu_{\\min}^{\\mathrm{small}}=%.6f$, $\\mu_{\\min}^{\\mathrm{far}}=%.6f$; $\\delta_{\\mathrm{cert}}(\\sigma_0)=%.6f$\n" % (mu_small_min, muL_min, delta_cert))
                f.write("\\end{itemize}\n")
            print(f"\nWrote LaTeX snippet to: {args.emit_tex}")
        except Exception as e:
            print(f"\n[warn] Failed to write LaTeX snippet: {e}")

    # ===== Per-σ covering budgets and margins =====
    def S_primes_upto_Q(exp: float, Q: int) -> float:
        return sum_primes_upto(Q, exp) if Q and Q > 0 else 0.0

    def K_sigma(s: float, Q: int, pmin_val: int, Cwin_val: float) -> float:
        # K(σ) := log p_min + (S_{σ+1/2}(Q)·log weights term)/(S_{σ+1/2}(Q)) ≈ log p_min + weighted avg(log q)
        # Use conservative majorant: log p_min + S_{σ+1/2}(Q,log)/S_{σ+1/2}(Q)
        a = s + 0.5
        # compute S_a(Q) and S_a_log(Q)
        small_primes = sieve_primes_upto(Q)
        S_a = sum((p ** (-a)) for p in small_primes)
        S_a_log = sum((p ** (-a)) * math.log(p) for p in small_primes)
        term_small = (S_a_log / S_a) if S_a > 0.0 else 0.0
        return math.log(max(pmin_val, 2)) + term_small

    if args.cov_sigma_start > 0.0 and args.Q and args.Q > 0:
        sigma_start = args.cov_sigma_start
        sigma_end = args.cov_sigma_end if args.cov_sigma_end > 0.0 else max(0.5005, sigma_start - 0.1)
        if sigma_start <= sigma_end:
            raise ValueError("cov_sigma_start must be > cov_sigma_end")
        # Choose p_min
        pmin_val = args.pmin if args.pmin and args.pmin > 0 else next_prime(args.Q)
        rows = []
        L_cum = 0.0
        sigma_k = sigma_start
        step_index = 0
        while sigma_k > sigma_end + 1e-12 and step_index < 10000:
            # adapt pmin to meet margin_min if needed
            trial_pmin = pmin_val
            # inner function to compute per-σ row with given pmin
            def compute_row_for_pmin(pmin_use: int):
                K_loc = K_sigma(sigma_k, args.Q, pmin_use, args.Cwin)
                h_by_theta = 0.9 * args.theta_max / max(K_loc, 1e-16)
                h_loc = min(args.h_max, h_by_theta)
                if sigma_k - h_loc < sigma_end:
                    h_loc = sigma_k - sigma_end
                theta_loc = K_loc * h_loc
                deltaL_loc = -math.log(max(1.0 - theta_loc, 1e-16))
                L_next = L_cum + deltaL_loc
                # budgets
                sigma_star = sigma_k + 0.5
                S_sigstar_Q = S_primes_upto_Q(sigma_star, args.Q)
                tail_majorant = ((pmin_use - 1) ** (1.0 - sigma_star)) / (sigma_star - 1.0) if sigma_star > 1.0 else float('inf')
                C4 = args.Cwin / 4.0
                pmin_sigstar = pmin_use ** (-sigma_star)
                Delta_FS = C4 * pmin_sigstar * S_sigstar_Q
                Delta_FF = C4 * pmin_sigstar * tail_majorant
                small_primes_loc = sieve_primes_upto(args.Q)
                L_values_small = [(p, (1.0 - sigma_k) * math.log(p) * (p ** (-sigma_k))) for p in small_primes_loc if p <= args.Q]
                L_max_small = max((L for p, L in L_values_small), default=0.0)
                mu_small_min = 1.0 - L_max_small / 6.0
                L_pmin = (1.0 - sigma_k) * math.log(pmin_use) * (pmin_use ** (-sigma_k))
                mu_far_min = 1.0 - L_pmin / 6.0
                two_term = (2 ** (-sigma_star)) if args.Q >= 2 else 0.0
                sum_small_excl2 = max(S_sigstar_Q - (2 ** (-sigma_star)), 0.0) if args.Q >= 2 else S_sigstar_Q
                Delta_SS = C4 * two_term * sum_small_excl2
                Delta_SF = C4 * two_term * tail_majorant
                delta_small_rows = mu_small_min - (Delta_SS + Delta_SF)
                delta_far_rows = mu_far_min - (Delta_FS + Delta_FF)
                delta_cert_loc = min(delta_small_rows, delta_far_rows)
                exp_neg_L = math.exp(-L_next)
                margin_loc = delta_cert_loc - exp_neg_L
                return (K_loc, h_loc, theta_loc, deltaL_loc, L_next,
                        delta_cert_loc, exp_neg_L, margin_loc,
                        Delta_FS, Delta_FF, Delta_SS, Delta_SF,
                        S_primes_upto_Q(sigma_k, args.Q), S_sigstar_Q, L_pmin, mu_far_min, mu_small_min)

            K, h, theta, delta_L, L_next, delta_cert_sigma, exp_neg_L, margin, Delta_FS, Delta_FF, Delta_SS, Delta_SF, S_sigma_Q, S_sigstar_Q, L_pmin, mu_far_min, mu_small_min = compute_row_for_pmin(trial_pmin)
            # adapt pmin upward if margin constraint requested
            while args.margin_min > 0.0 and margin < args.margin_min and trial_pmin < args.pmin_max:
                trial_pmin = next_prime(trial_pmin)
                K, h, theta, delta_L, L_next, delta_cert_sigma, exp_neg_L, margin, Delta_FS, Delta_FF, Delta_SS, Delta_SF, S_sigma_Q, S_sigstar_Q, L_pmin, mu_far_min, mu_small_min = compute_row_for_pmin(trial_pmin)
            pmin_val = trial_pmin
            # θ bound with a safety factor 0.9 to stay within theta_max
            L_cum = L_next
            rows.append({
                'k': step_index + 1,
                'sigma': sigma_k,
                'h': h,
                'K_sigma': K,
                'theta': theta,
                'delta_L': delta_L,
                'L_cum': L_cum,
                'exp_neg_L': exp_neg_L,
                'delta_cert': delta_cert_sigma,
                'margin': margin,
                'Delta_FS': Delta_FS,
                'Delta_FF': Delta_FF,
                'Delta_SS': Delta_SS,
                'Delta_SF': Delta_SF,
                'S_sigma_Q': S_sigma_Q,
                'S_sigstar_Q': S_sigstar_Q,
                'pmin': pmin_val,
                'L_pmin': L_pmin,
                'mu_far': mu_far_min,
                'mu_small': mu_small_min,
            })
            sigma_k -= h
            step_index += 1

        # Emit CSV if requested
        if args.emit_cover_csv:
            try:
                with open(args.emit_cover_csv, 'w', encoding='utf-8') as f:
                    f.write("k,sigma_k,h_k,K_sigma,theta_k,L_step,L_cum,exp_neg_L,delta_cert,margin,DFS,DFF,DSS,DSF,S_sigma,S_sigma_half,pmin,L_pmin,mu_far,mu_small\n")
                    for r in rows:
                        f.write(
                            f"{r['k']},{r['sigma']:.6f},{r['h']:.6f},{r['K_sigma']:.6f},{r['theta']:.6f},{r['delta_L']:.6f},{r['L_cum']:.6f},{r['exp_neg_L']:.6f},{r['delta_cert']:.6f},{r['margin']:.6f},{r['Delta_FS']:.6f},{r['Delta_FF']:.6f},{r['Delta_SS']:.6f},{r['Delta_SF']:.6f},{r['S_sigma_Q']:.6f},{r['S_sigstar_Q']:.6f},{r['pmin']},{r['L_pmin']:.6f},{r['mu_far']:.6f},{r['mu_small']:.6f}\n"
                        )
                print(f"Wrote covering CSV: {args.emit_cover_csv}")
            except Exception as e:
                print(f"[warn] Failed to write covering CSV: {e}")

        # Emit LaTeX table if requested
        if args.emit_cover_tex:
            try:
                with open(args.emit_cover_tex, 'w', encoding='utf-8') as f:
                    f.write("% Auto-generated per-σ covering table\n")
                    f.write("\\begin{table}[H]\n\\centering\n")
                    f.write("\\caption{Per-$\\sigma$ covering: $Q=%d$, $p_{\\min}=%d$, $C_{\\mathrm{win}}=%.2f$, $\\theta_{\\max}=%.2f$, $h_{\\max}=%.3f$.}\n" % (args.Q, pmin_val, args.Cwin, args.theta_max, args.h_max))
                    f.write("\\small\n\\begin{tabular}{r r r r r r r r r}\\toprule\n")
                    f.write("$k$ & $\\sigma_k$ & $h_k$ & $K(\\sigma_k)$ & $\\theta_k$ & $L(\\sigma_k)$ & $e^{-L}$ & $\\delta_{\\rm cert}$ & margin \\\\ \\midrule\n")
                    for r in rows:
                        f.write(f"{r['k']} & {r['sigma']:.4f} & {r['h']:.4f} & {r['K_sigma']:.6f} & {r['theta']:.6f} & {r['L_cum']:.6f} & {r['exp_neg_L']:.6f} & {r['delta_cert']:.6f} & {r['margin']:.6f} \\\\\n")
                    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
                print(f"Wrote covering LaTeX: {args.emit_cover_tex}")
            except Exception as e:
                print(f"[warn] Failed to write covering LaTeX: {e}")


if __name__ == "__main__":
    main()


