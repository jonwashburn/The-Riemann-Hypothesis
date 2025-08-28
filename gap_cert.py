# gap_cert.py  —  finite-block spectral-gap certificate (stdlib only)
# Implements off-diagonal block norms, interval Gershgorin, and two gap bounds.

from dataclasses import dataclass
from math import sqrt, pow
from typing import List, Tuple, Dict, Optional

Interval = Tuple[float, float]                 # [lo, hi]
IntervalMatrix = List[List[Interval]]          # N x N interval matrix


@dataclass(frozen=True)
class Block:
    """A single prime block."""
    p: int                      # the prime
    N: int                      # block size N_p
    D_interval: IntervalMatrix  # intervals for D̃_p(σ), σ ∈ [σ0, 1]
    weight_sum: Optional[float] = None  # optional override for ∑ w_i


# ---------- helpers for interval Gershgorin ----------

def sup_abs(lo: float, hi: float) -> float:
    """Supremum of |x| on [lo, hi]."""
    return max(abs(lo), abs(hi))


def gershgorin_mu_lower(D: IntervalMatrix) -> float:
    """
    Interval Gershgorin: uniform lower bound μ_p^L for the smallest eigenvalue of D over σ∈[σ0,1].
    Assumes D[i][i] = [dlo, dhi] encloses the diagonal for all σ and
    off-diagonals D[i][j] = [lo, hi] enclose entries for all σ.
    """
    N = len(D)
    assert all(len(row) == N for row in D), "D must be square"
    mu_candidates = []
    for i in range(N):
        dlo, _dhi = D[i][i]
        diag_lower = dlo
        row_sum = 0.0
        for j in range(N):
            if i == j:
                continue
            lo, hi = D[i][j]
            row_sum += sup_abs(lo, hi)
        mu_candidates.append(diag_lower - row_sum)
    return min(mu_candidates) if mu_candidates else 0.0


# ---------- off-diagonal block budgets U_pq ----------

def weight_sum_unweighted(N: int) -> float:
    """For the unweighted model, the Upq bound uses √(N_p N_q)."""
    return sqrt(max(N, 0))


def weight_sum_weighted(N: int) -> float:
    """
    For weights w_n = 3^{-(n+1)},  n=0..N-1.
    Sum is (1/2)*(1 - 3^{-N}) ≤ 1/2.
    """
    if N <= 0:
        return 0.0
    return 0.5 * (1.0 - pow(3.0, -N))


def Upq_upper(p: int, q: int, wp: float, wq: float,
              sigma0: float, Cwin: float = 1.0, slack: float = 1e-15) -> float:
    """
    Bound:  U_pq ≤ Cwin * wp * wq * (p q)^{-σ0}.
    `slack` inflates result slightly to preserve '≤' under roundoff.
    """
    val = Cwin * wp * wq * pow(p * q, -sigma0)
    return val * (1.0 + slack)


def S_sigma0_integer_tail(P: int, sigma0: float) -> float:
    """
    Trivial integer-tail bound:
    S_{σ0}(P) := ∑_{n≤P} n^{-σ0} ≤ 1 + (P^{1-σ0} - 1)/(1-σ0),  for σ0 ∈ (1/2, 1).
    """
    if not (0.5 < sigma0 < 1.0):
        raise ValueError("This bound assumes σ0 ∈ (1/2, 1).")
    return 1.0 + (pow(P, 1.0 - sigma0) - 1.0) / (1.0 - sigma0)


# ---------- budgets: exact (finite sum over given primes) or tail-bound ----------

def _weights_for_blocks(blocks: List[Block], model: str) -> Dict[int, float]:
    w: Dict[int, float] = {}
    for b in blocks:
        if model == 'unweighted':
            w[b.p] = weight_sum_unweighted(b.N)
        elif model == 'weighted':
            w[b.p] = weight_sum_weighted(b.N) if b.weight_sum is None else b.weight_sum
        else:
            raise ValueError("model must be 'unweighted' or 'weighted'")
    return w


def offdiag_budget_exact(blocks: List[Block], sigma0: float, model: str,
                         Cwin: float, slack: float):
    """
    Returns:
      Upq_sums[p] = ∑_{q≠p} U_pq  (exact finite sum over the provided prime list),
      Upq_mat[(p,q)] = U_pq (for Schur bound),
      w[p] = weight factor per block.
    """
    w = _weights_for_blocks(blocks, model)
    Upq_sums: Dict[int, float] = {b.p: 0.0 for b in blocks}
    Upq_mat: Dict[Tuple[int, int], float] = {}
    for bp in blocks:
        for bq in blocks:
            if bp.p == bq.p:
                continue
            u = Upq_upper(bp.p, bq.p, w[bp.p], w[bq.p], sigma0, Cwin=Cwin, slack=slack)
            Upq_mat[(bp.p, bq.p)] = u
            Upq_sums[bp.p] += u
    return Upq_sums, Upq_mat, w


def offdiag_budget_tail(blocks: List[Block], sigma0: float, model: str,
                        Cwin: float, P: int, Nmax: Optional[int]):
    """
    Tail budget using explicit finite-P inequality.
    For each block p:
      unweighted: ∑_{q≠p} U_pq ≤ Cwin * Nmax * p^{-σ0} * S_{σ0}(P)
      weighted:   ∑_{q≠p} U_pq ≤ Cwin * (1/4) * p^{-σ0} * S_{σ0}(P)
    Returns Upq_sums[p].
    """
    S = S_sigma0_integer_tail(P, sigma0)
    Upq_sums: Dict[int, float] = {}
    for b in blocks:
        if model == 'unweighted':
            if Nmax is None:
                raise ValueError("Nmax is required for unweighted tail budget.")
            Upq_sums[b.p] = Cwin * float(Nmax) * pow(b.p, -sigma0) * S
        elif model == 'weighted':
            Upq_sums[b.p] = Cwin * 0.25 * pow(b.p, -sigma0) * S
        else:
            raise ValueError("model must be 'unweighted' or 'weighted'")
    return Upq_sums


# ---------- main gap wrapper ----------

def gap_lower_bound_sigma0(
    sigma0: float,
    blocks: List[Block],
    *,
    model: str = 'unweighted',       # 'unweighted' or 'weighted'
    Cwin: float = 1.0,
    budget: str = 'exact',            # 'exact' (sum over given primes) or 'tail' (use bound)
    P: Optional[int] = None,          # needed for budget='tail'
    Nmax: Optional[int] = None,       # needed for unweighted tail mode
    slack: float = 1e-15              # small inflation to preserve '≤' under float roundoff
):
    """
    Computes and returns a certified lower bound δ(σ0):

      δ(σ0) ≥ max{ 0,
                   min_p( μ_p^L  −  ∑_{q≠p} U_pq ) ,
                   min_p μ_p^L  −  max_q ∑_{p≠q} √μ_p^L U_pq / √μ_q^L }.

    μ_p^L is obtained by interval Gershgorin on the provided D̃_p(σ) intervals.

    Returns a dict with:
      - 'delta', 'delta1', 'delta2'
      - per-block 'muL', 'Upq_sums', and (for exact budgets) 'Upq_mat' + 'weights'.
    """
    # 1) Certified μ_p^L (interval Gershgorin)
    muL: Dict[int, float] = {}
    for b in blocks:
        mu = gershgorin_mu_lower(b.D_interval)
        muL[b.p] = mu

    # 2) Off-diagonal budgets
    if budget == 'exact':
        Upq_sums, Upq_mat, weights = offdiag_budget_exact(blocks, sigma0, model, Cwin, slack)
    elif budget == 'tail':
        if P is None:
            raise ValueError("P is required for budget='tail'.")
        Upq_sums = offdiag_budget_tail(blocks, sigma0, model, Cwin, P, Nmax)
        Upq_mat, weights = {}, _weights_for_blocks(blocks, model)
    else:
        raise ValueError("budget must be 'exact' or 'tail'.")

    # 3) δ1: blockwise diagonal dominance (Gershgorin at block-level)
    delta1 = min(muL[p] - Upq_sums[p] for p in muL) if muL else 0.0

    # 4) δ2: Schur-type bound (needs exact Upq_mat and positive μ_p^L)
    if budget == 'exact' and muL and all(muL[p] > 0.0 for p in muL):
        worst_sum = float('-inf')
        for bq in blocks:
            sq = 0.0
            for bp in blocks:
                if bp.p == bq.p:
                    continue
                u = Upq_mat[(bp.p, bq.p)]
                sq += ( (muL[bp.p]**0.5) * u / (muL[bq.p]**0.5) )
            if sq > worst_sum:
                worst_sum = sq
        delta2 = min(muL.values()) - worst_sum
    else:
        delta2 = float('-inf')  # unavailable / not applicable

    # 5) Final certified gap
    delta = max(0.0, delta1, delta2)

    out = {
        'delta':  delta,
        'delta1': delta1,
        'delta2': delta2,
        'muL':    muL,
        'Upq_sums': Upq_sums,
        'weights':  weights,
        'model':    model,
        'Cwin':     Cwin,
        'sigma0':   sigma0,
        'budget':   budget,
        'slack':    slack,
    }
    if budget == 'exact':
        out['Upq_mat'] = Upq_mat
    return out


