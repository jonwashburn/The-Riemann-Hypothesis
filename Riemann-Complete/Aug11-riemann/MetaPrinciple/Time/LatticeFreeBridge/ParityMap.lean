/-
  MetaPrinciple/Time/LatticeFreeBridge/ParityMap.lean
  Minimal parity bridge: just enough for no_cover_with_period_lt_eight'
-/

namespace MetaPrinciple.Time.LatticeFreeBridge

/-- Concrete state space: 3 Boolean parities. -/
abbrev V : Type := Fin 3 → Bool

/-- The parity map is literally the identity. -/
@[simp] def π : V → (Fin 3 → Bool) := id

@[simp] theorem π_apply (v : V) : π v = v := rfl

/-- Surjectivity of the parity map (trivial because π = id). -/
theorem hπ : Function.Surjective π :=
  fun p => ⟨p, rfl⟩


/-! -------------------------------------------------------------
     Optional: d‑bit generalization you can reuse elsewhere
    ------------------------------------------------------------- -/

/-- d‑bit state space. -/
abbrev Vd (d : Nat) : Type := Fin d → Bool

/-- d‑bit parity map (also the identity). -/
@[simp] def πd (d : Nat) : Vd d → (Fin d → Bool) := id

/-- Surjectivity of the d‑bit parity map. -/
theorem hπd (d : Nat) : Function.Surjective (πd d) :=
  fun p => ⟨p, rfl⟩

end MetaPrinciple.Time.LatticeFreeBridge
