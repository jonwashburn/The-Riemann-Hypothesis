import MetaPrinciple.LatticeFree

open Function

namespace MetaPrinciple
namespace Time

/-- Lattice‑independent reformulation of the eight‑tick minimality.
Given any state space `V` equipped with a surjective parity map into the
3‑bit pattern space, no surjection from `Fin T` onto `V` exists when `T < 8`.

Use this with your concrete `V` (e.g. voxel states or recognition states)
by providing `π : V → (Fin 3 → Bool)` and a proof it is surjective. -/
theorem no_cover_with_period_lt_eight'
  {V : Type}
  (π : V → (Fin 3 → Bool)) (hπ : Surjective π) :
  ∀ {T : Nat}, T < 8 → ¬ ∃ f : Fin T → V, Surjective f := by
  intro T hT h
  rcases h with ⟨f, hf⟩
  -- parity coverage is surjective by composition
  have hcov : Surjective (fun i : Fin T => π (f i)) := hπ.comp hf
  -- contradict the lattice‑free bound
  have : 8 ≤ T := LatticeFree.eight_tick_min (pass := fun i => π (f i)) hcov
  exact (not_lt_of_ge this) hT

end Time
end MetaPrinciple
