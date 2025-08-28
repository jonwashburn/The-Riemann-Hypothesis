## PSC Manuscript Punchlist (Aug-18-riemann-corrected.tex)

Status: actionable fixes to ensure the PSC route is unconditional and referee-proof.

1) Remove any remaining global-BMO machinery
- Delete every occurrence of: “Fefferman–Stein”, “Lusin area”, “John–Nirenberg”.
- Ensure Hilbert pairing sections explicitly state no global BMO is used.

2) Print the corrected arithmetic constant K0
- Ensure the lemma states: K0 := (1/4) \sum_p \sum_{k\ge2} p^{-k}/k^2 < \infty and \iint_{Q(I)} |\nabla U_{det2}|^2\,\sigma \le K0 |I|.

3) Whitney scale declared and used in ξ–energy
- Explicitly declare L(T) := c / log⟨T⟩ in the ξ–energy lemma.
- Use it to turn far-annulus sums into O(L) and near-zero counts into O(1).

4) State and use unconditional short-interval zero count
- Include: N(T;H) \ll A0 + A1 (H log T + log T), used directly in ξ–energy near/far analysis.

5) Certificate constants printed and inequality displayed
- Print c0(ψ) = (arctan 2)/(2π) and the inequality: (C_H(ψ) M_ψ + C_P(κ)) / c0(ψ) < π/2, with a numeric evaluation for the printed window.

6) Remove/downgrade uniform L1-in-ε claims
- Delete or rewrite any “uniform L1” convergence statements; keep only smoothed/distributional results that are proved.

7) Define A(ψ) where used
- Display: A(ψ)^2 := ∬_{R^2_+} |∇(P_σ * ψ)|^2 σ dt dσ.

8) Cross-term control in ξ–energy (near and far)
- Near: record the (N_near)^2 L bound after Cauchy–Schwarz.
- Far: retain the Schur kernel bound sup_γ Σ_{γ′} K(γ,γ′) \lesssim L^2 log⟨T⟩ 2^{-k} + L 4^{-k} and sum in k.

9) Whitney-to-global reduction
- Keep the full partition-of-unity proof reducing sup over all (L,t0) to Whitney scales for both M_ψ and C_H(ψ).

10) Localized pairing with full boundary terms
- Ensure the cutoff pairing identity on Q(α′I) includes and bounds side/top remainder terms.

Verification
- Rebuild with pdflatex; grep confirms no global BMO terms; certificate constants present; A(ψ) defined; Whitney-scale and short-interval bounds in ξ–energy.


