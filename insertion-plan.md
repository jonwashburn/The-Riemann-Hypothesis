# Integration Plan — Truth‑First Assessment Remediation

Purpose: Organize and track insertion of the new, completed components you’ll provide (one by one) to close the editorial requirements. Each item below lists: (a) what we need, (b) expected file/snippet name, (c) where to insert (anchor order in `riemann-aug-21-ten00pm.tex`), (d) dependencies/labels, and (e) acceptance criteria.

We will update STATUS and PREREQS as we integrate each artifact and re‑build.

---

## 0) Current foundation already in manuscript (for context)
- boundary regularity + outer normalization + distributional phase–velocity: `boundary-regularity.tex` (inserted)
- phase–balayage (test and interval forms) + neutralization: `phase-balayage.tex` (inserted)
- wedge lemma (two‑sided oscillation ⇒ wedge): `wedge-lemma.tex` (inserted)
- ξ–block Carleson boxes (unconditional K_ξ): `carleson-boxes.tex` (inserted)
- fixed–aperture H^1–BMO / Carleson embedding (M_ψ bound): `carleson-embedding.tex` (inserted)
- height uniformity + pinch/exhaustion: `uniformity-in-T.tex` (inserted)
- proof-map summary (orientation only): `proof-map-snippet.tex` (inserted)

---

## 1) Phase–Certificate Theorem (the linchpin)
- WHAT: A single theorem that states and proves
  \[ \Upsilon:=\frac{C_H(\psi)\,M_\psi + C_P(\kappa)}{c_0(\psi)}\ \le\ \tfrac12\ \Rightarrow\ w\in[-\tfrac{\pi}{2},\tfrac{\pi}{2}]\ \text{a.e. on }\partial\Omega. \]
  with explicit hypotheses, boundary trace notions, function‑space class of `2\mathcal J`, and constants.
- FILE: `phase-certificate-theorem.tex`
- INSERT AFTER: `wedge-lemma.tex`
- DEPENDENCIES: `boundary-regularity.tex`, `phase-balayage.tex`, `wedge-lemma.tex`, `carleson-embedding.tex`, definition of `c_0, C_H, C_P, C_ψ^{(H^1)}`.
- LABELS: `thm:phase-certificate`
- ACCEPTANCE: Proof self‑contained; constants/normalizations fixed; references only local inserts.
- STATUS: PENDING

## 2) Unconditional, uniform C_box bound — ξ‑block (no circularity)
- WHAT: Full write‑up with constants for K_ξ; worst‑case configuration; Whitney scaling uniformity.
- FILE: (already present) `carleson-boxes.tex`
- INSERT AFTER: `phase-balayage.tex` and `wedge-lemma.tex` (already done)
- DEPENDENCIES: none beyond standard zero‑count and local neutralizer; labels `lem:cubic`, `thm:Kxi`
- ACCEPTANCE: No log⟨T⟩ leak; normalized O(|I|) bound uniform in height.
- STATUS: DONE

## 3) Fixed–aperture H^1–BMO / Carleson embedding with constants
- WHAT: Self‑contained statement/proof under exact normalization; bound `M_ψ ≤ (4/π) C_ψ^(H^1) √C_box`.
- FILE: (already present) `carleson-embedding.tex`
- INSERT AFTER: `carleson-boxes.tex` (done)
- DEPENDENCIES: window class `ψ`, aperture `α`, Poisson normalization.
- ACCEPTANCE: Constants tracked end‑to‑end; aperture dependence explicit.
- STATUS: DONE

## 4) Removability at Z(ξ) (clean statement)
- WHAT: Short lemma(s) showing bounded `\Theta` near a puncture ⇒ holomorphic extension; exclude `\Theta(ρ)=1`; conclude `\mathcal J` has no pole.
- FILE: `boundary-to-interior.tex` (or include in `uniformity-in-T.tex`)
- INSERT AFTER: `uniformity-in-T.tex` (if separated) or verify existing Section covers it.
- DEPENDENCIES: `\Theta` holomorphic on rectangles; Cayley relation.
- ACCEPTANCE: Explicit Riemann removable singularity argument; maximum‑modulus note.
- STATUS: PENDING (covered in `uniformity-in-T.tex` but will upgrade if you provide a standalone)

## 5) Globalization (“pinch”) formal lemma
- WHAT: Overlapping rectangles scheme; two‑constants inequality; no loss of constants; normal‑families variant.
- FILE: (present) `uniformity-in-T.tex`
- INSERT ORDER: After embedding; before proof map (done)
- ACCEPTANCE: Explicit statements/labels `lem:two-const-*`, `thm:pinch-*` with proofs.
- STATUS: DONE

## 6) Single closed chain to RH (product certificate route)
- WHAT: Ensure the manuscript presents one complete chain; PSC/density archived or clearly marked non‑essential.
- FILE: Edit short orientation paragraph or appendix, if needed.
- INSERT: Proof‑map or intro section.
- STATUS: PENDING (ready to adjust once all inserts are in)

## 7) Consolidated numerics with error control
- WHAT: One appendix consolidating locked constants (K_0, K_ξ, \|U_Γ\|, C_ψ^(H^1), C_H, C_P, c_0) with monotone tails/enclosures.
- FILE: `appendix-constants.tex`
- INSERT: At end, before `proof-map-snippet.tex`.
- STATUS: PENDING (we’ll keep the existing appendices or consolidate per your artifact)

---

## Insertion order (anchor sequence in main file)
1) `boundary-regularity` (inserted)
2) `phase-balayage` (inserted)
3) `wedge-lemma` (inserted)
4) `phase-certificate-theorem` (to be inserted here)
5) `carleson-boxes` (inserted)
6) `carleson-embedding` (inserted)
7) `uniformity-in-T` (inserted)
8) `appendix-constants` (optional consolidation)
9) `proof-map-snippet` (inserted)

---

## Label & macro conventions
- Theorems: `thm:*`, Lemmas: `lem:*`, Props: `prop:*`, Cor: `cor:*` with section‑specific suffixes.
- Use existing macros for Poisson/Hilbert transforms and window constants.
- Avoid redefining packages/macros in snippets; rely on master preamble.

---

## Build & verification checklist (per insertion)
- [ ] `pdflatex -interaction=nonstopmode -halt-on-error riemann-aug-21-ten00pm.tex`
- [ ] Search for duplicate `\end{document}` or repeated `\input{...}`
- [ ] Grep for unresolved labels: `rg "Reference.*undefined|Label.*multiply defined" riemann-aug-21-ten00pm.log`
- [ ] Quick scan: certificate inequality references (c_0, C_H, C_P, C_ψ^(H^1), C_box)

---

## Working notes
- As you send each completed component, I’ll drop it into the matching file name above, update STATUS here, add the `\input{...}` at the anchor, and rebuild.
- If your artifact already overlaps an inserted section, we’ll either replace the existing snippet or merge content under the same labels.
