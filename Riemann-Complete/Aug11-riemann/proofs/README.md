# Proof Attempts Registry

This folder tracks multiple LaTeX proof attempts toward RH with clear blockers encoded in filenames and summaries.

## Naming convention
- attempt_YYYY-MM-DD_<tag>.tex
- Tags highlight the distinguishing choices and the open blocker.

## Current attempts

- attempt_2025-08-11_det3_kernel_psd_hypothesis.tex
  - Regularization: det_3 with E_3(s) = exp(∑ p^{-s} + 1/2 ∑ p^{-2s}).
  - Kernel: K = (1 - Θ Θ̄)/(Σ + Σ̄) with Σ(s)=∑ p^{-2}/(1-p^{-s}).
  - BLOCKER: Prove Θ is Schur relative to Σ on Re s > 1/2 (equivalently, K is PSD). Under this, RH follows.

- attempt_2025-08-11_opslog_det2_schur_condition.tex
  - Ops‑log styled version aligning with det₂ bookkeeping and constraints (M1–M3).
  - BLOCKER: Same Schur/PSD statement for (Θ, Σ) on Re s > 1/2.

- attempt_2025-08-11_adelic_finite_truncation_regularization.tex
  - Adelic operator + finite truncation regularization perspective.
  - BLOCKER: Finite‑N PSD ⇒ limit PSD for K together with realization; stated as a finite‑to‑infinite PSD hypothesis.

## How to compile
From this folder (or project root):

```bash
latexmk -pdf attempt_2025-08-11_det3_kernel_psd_hypothesis.tex
latexmk -pdf attempt_2025-08-11_opslog_det2_schur_condition.tex
latexmk -pdf attempt_2025-08-11_adelic_finite_truncation_regularization.tex
```

If latexmk is unavailable, use pdflatex twice.

## Notes
- All unconditional lemmas and standard functional‑analytic steps are included in each attempt.
- Each file isolates its single core hypothesis (BLOCKER) so progress can be tracked per approach.
