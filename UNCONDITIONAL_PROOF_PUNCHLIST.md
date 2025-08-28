# Unconditional RH Proof – Punchlist (riemann-verified-complete.tex)

Goal: make the manuscript self‑contained, unconditional, and audit‑complete for the boundary route (PSC ⇒ (P+) ⇒ Herglotz/Schur ⇒ RH). We keep conditional variants in an appendix, clearly labeled.

## A. Immediate blockers (must fix)

- [ ] Abstract: updated to state boundary closure via certificate (done)
- [ ] Eliminate any remaining phrasing that treats (P+) as a hypothesis; consistently cite the proved certificate (Theorem `unconditional-choice`).
- [ ] Ensure the boundary‑positivity drop‑in is loaded where needed (done) and harmonized with main text labels.
- [ ] Resolve all cross‑references (`\ref{…}`) – no missing labels or dangling refs.
  - [ ] `prop:phase-velocity-identity` (phase–velocity identity): provide precise statement + proof, or retarget to the in‑text proposition that now contains the identity (currently Proposition after line ~1417 / 1423).
  - [ ] `lem:outer-factorization` (outer phase = Hilbert transform of log‑modulus) – add a lemma with full proof or cite a standard reference with exact hypotheses.
  - [ ] `lem:HS-variation`, `app:lipschitz`, `cor:global-schur`, `lem:compact-alignment`, `lem:nonvanish-det2`, `thm:psc-calibrated` – either add these or retarget to existing, nearby results.
- [ ] Remove residual “Under RH” phrasing and make the boundary route argument rely only on (P+) proved by Theorem `unconditional-choice` and Theorem `global-PSD`.
- [ ] Replace remaining “proof sketch” markers with full proofs or precise citations in the smoothed/derivative lemmas.
- [ ] Add a short bibliography (Titchmarsh; Simon; de Branges–Rovnyak/Garnett; Edwards/Iwaniec–Kowalski) and wire citations in-text.

## B. Boundary route – proof completeness

- [ ] Phase–velocity identity
  - [ ] State identity precisely (function classes, normalization, distributional sense, removal of critical‑line atoms on short intervals).
  - [ ] Prove it fully (factorization F = inner × outer; outer phase = H[u']; Blaschke/atoms for off‑line poles and critical‑line zeros; testing against nonnegative φ).

- [ ] Outer normalization
  - [ ] Fully prove Theorem `uniform-eps` (already mostly present). Ensure the de‑smoothing chain and smoothed bounds have complete statements and justified constant choices.
  - [ ] Add the lemma “outer phase = Hilbert transform of log‑modulus” with hypotheses matching our outer construction.

- [ ] Certificate theorem (Section `sec:certificate`)
  - [ ] Verify definitions: `C_Γ^{(L)}`, `C_P(ψ,L)`, `C_H(ψ,L)`, `c_0(ψ)` are all present and consistent with the phase–velocity decomposition used earlier (prime–det2 difference for `C_P`).
  - [ ] Provide a full proof of the certificate inequality `(⋆)` using the earlier lemmas (no “sketch”).
  - [ ] Ensure scale reduction lemma for `c_0(ψ)` is included and correct.

- [ ] Unconditional parameter choice (Theorem `unconditional-choice`)
  - [ ] Expand the proof to a full argument: explicitly bound each term on the adaptive cover and show `C*/c0 ≤ π/2` uniformly.
  - [ ] Check constants are defined once: `A_ψ := C_Γ(ψ) + C_H(ψ)`; `κ := π c0(ψ)/8`; `c := π c0(ψ)/(8 A_ψ)`.
  - [ ] Confirm all prerequisites are in this manuscript and labeled.

## C. Interior route – clarity and scope

- [ ] Keep interior results as local/off‑zeros; note that global Schur is now delivered by the boundary route.
- [ ] Retain the `$k=1$ blocker` remark; ensure it’s clearly orthogonal to the unconditional boundary proof.
- [ ] AFK lift cleanup (Lemma/Section around AFK features): remove the incorrect identification `E_N = J_N`; keep `E_N(s,\bar t)` as the Fock–Gram kernel and use the proven PSD inequality `K_{exp,N} - K_{FG,N} \succeq 0` plus congruence by `\xi^{-1}` (see Proposition `prop:boundary-psd-fixed`). Ensure no place equates the kernel to the scalar ratio `J_N(s)`.
- [ ] Replace the line asserting `E_N = \exp(\Lambda_N) = J_N` with a reference to the boundary PSD transfer (Proposition `prop:boundary-psd-fixed`); verify the AFK section only builds the Gram pieces and never identifies the kernel with `J_N` itself.

## D. Conditional routes – move to appendix / clearly marked

- [ ] Littlewood‑type bounds ⇒ (P+) ⇒ RH: keep in a “Conditional variants” appendix and mark as optional.
- [ ] Conjectural short‑interval Poisson mass bound ⇒ (P+) ⇒ RH: keep as a conjecture/roadmap.

## E. Cross‑reference & label audit

- [ ] Search for all occurrences of “Boundary hypothesis (P+)” and replace with “Boundary positivity (P+) (proved via Theorem `unconditional-choice`).”
- [ ] Confirm every `\ref{…}` resolves; add missing labels or adjust citations accordingly.
- [ ] Ensure the drop‑in labels (`def:Pplus`, `thm:Pplus-balayage`, etc.) do not clash with existing labels.

## F. Bibliography / standard facts

- [ ] Add a minimal bibliography (Titchmarsh for Γ/ζ bounds; Simon for trace ideals; de Branges–Rovnyak / Garnett for outer/Schur; Edwards/Iwaniec–Kowalski for explicit formula).
- [ ] Where we quote classical inequalities (digamma/Stirling, Poisson/Herglotz), provide inline references or short proofs.

## G. Build/test

- [ ] Compile the LaTeX; fix warnings and undefined references.
- [ ] Quick consistency QA: every theorem referenced appears earlier or is fully restated; no “…” placeholders remain.
- [ ] Final pass to ensure “unconditional” appears only where we directly proved the claim.

---

### Proposed edit order
1) Add/retarget precise phase–velocity identity + outer‑phase lemma.  
2) Complete proofs in Section `sec:certificate` and Theorem `unconditional-choice`.  
3) Sweep and fix all refs/phrasings re: (P+).  
4) Move conditional routes to Appendix; add short bibliography.  
5) Compile and resolve any remaining issues.


