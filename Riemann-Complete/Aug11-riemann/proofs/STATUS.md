# Proof Tracks — Status and Next Actions

This file summarizes each proof attempt, its target, current blocker, and concrete next step.

## attempt_2025-08-11_det3_kernel_psd_hypothesis.tex
- Goal: PSD of `K(s,w)` for `(Θ=ζ^2/E3^{-1}, Σ)` using det_3 regularization; then dB–R realization ⇒ RH.
- Status: Conditional. PSD for the given `(Θ,Σ)` is unproven; diagonal test suggests failure for real s>1 if Θ not Schur.
- Blocker: Prove `Θ ∈ Schur(Σ)` on `Re s>1/2` (or equivalent kernel PSD). RH-strength.
- Next: Keep as a conditional record; do not pursue unless a new inequality is found.

## attempt_2025-08-11_opslog_det2_schur_condition.tex
- Goal: Formal ops-log conditional program with det_2 matching (M1–M3) and Schur condition.
- Status: Conditional. Same blocker: `Θ ∈ Schur(Σ)`.
- Blocker: Kernel positivity for current `(Θ,Σ)`.
- Next: Maintain as the clean conditional reference.

## attempt_2025-08-11_adelic_finite_truncation_regularization.tex
- Goal: Adelic operator + finite truncation regularization; pass PSD from finite N to limit.
- Status: Conditional. Finite PSD hypothesis (Keystone A) not established for current `Θ_N`.
- Blocker: Show each `K_N` is PSD or provide a different finite PSD construction that converges to the target `K`.
- Next: Keep as conditional; do not claim finite PSD without proof.

## attempt_2025-08-11_keystone_A.tex (and _revised)
- Goal: Keystone A finite-PSD → limit PSD → dB–R realization.
- Status: Revised version withdraws unproven claims; finite-PSD is marked as a hypothesis.
- Blocker: Prove finite-PSD for `K_N` (equiv., `Θ_N` Schur w.r.t. `Σ_N`).
- Next: None until a new finite-stage argument appears.

## attempt_2025-08-11_opslog_reconstructed_proof.tex
- Goal: Reconstructed, formal conditional proof from ops-log.
- Status: Conditional; clear single blocker noted.
- Blocker: `Θ ∈ Schur(Σ)` on `Re s>1/2`.
- Next: Keep as the canonical write-up.

## appendix_2025-08-11_FOT_kernel_operator_framework_single_lemma_gap.tex
- Goal: Self-contained FOT → kernel PSD → operator realization framework with a single explicit BLOCKER.
- Status: Complete as a framework; BLOCKER remains external.
- Blocker: Existence of a Σ–Schur `Θ` matching ζ-side determinant identity.
- Next: Use as appendix/reference.

## attempt_2025-08-11_schurized_contrast_cayley_psd.tex
- Goal: Unconditional PSD via Schurized contrast `˜Θ=(H−1)/(H+1)` for Herglotz `H`.
- Status: Unconditional PSD achieved; realization standard. Not yet tied to `ξ(s)`.
- Blocker: Determinant identity `det₂(I−T(s))=ξ(s)` for the passive colligation.
- Next: Prove existence of a passive colligation whose log-det reproduces the Euler product (prime block) and normalizers, yielding `ξ(s)` exactly.

## status_2025-08-11_RS-RH-KERNEL-STATUS_opslog.tex
- Goal: Ops-log snapshot of global status.
- Status: Up to date.
- Next: Update when a blocker moves.

---

## Global targets (unconditional only)
- T1: PSD kernel on `Re s>1/2` tied to ζ (not a modified function), or an alternative unconditional route yielding RH for ζ.
- T2: Passive (or J-conservative if required) realization with strict contractivity off `Re s=1/2`.
- T3: Exact determinant identity `det₂(I−T(s))=ξ(s)` and eigenvalue/zero equivalence on the same Hilbert (not modified) setup.

## Single critical lemma to close (current best route)
- LEMMA: Construct a passive colligation with characteristic function consistent with the Herglotz-based kernel such that `det₂(I−T(s))=ξ(s)` and `‖T(s)‖<1` on `Re s>1/2`.
- Impact: Converts the current conditional program into an unconditional proof.

