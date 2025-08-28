import MetaPrinciple.Time.LatticeFreeBridge
import MetaPrinciple.Time.LatticeFreeBridge.ParityMap

open Function
open MetaPrinciple.Time
open MetaPrinciple.Time.LatticeFreeBridge

/-- Minimal eight-tick theorem using the lattice-independent bridge
    and the identity parity map on `V = Fin 3 → Bool`. -/
theorem eight_min : ∀ {T : Nat}, T < 8 → ¬ ∃ f : Fin T → V, Surjective f :=
  no_cover_with_period_lt_eight' π hπ
