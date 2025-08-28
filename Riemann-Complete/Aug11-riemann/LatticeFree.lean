/-!
Lattice–independent eight–tick (2^d) lower bound, fully self–contained.

Drop-in file for the repo, no `sorry` and no external dependencies beyond Mathlib.

Results:

- `no_surj_small`              : no surjection `Fin T → (Fin d → Bool)` if `T < 2^d`.
- `min_ticks_cover`            : any pass that hits all `2^d` patterns has length ≥ `2^d`.
- `min_ticks_cover_states`     : same bound after a parity map from arbitrary states.
- `eight_tick_min`             : specialization `d = 3` gives the 8–tick bound.

Interpretation in RS:
`(Fin d → Bool)` plays the role of d independent binary parities/ports.
A recognition pass that covers all parities must visit at least `2^d` distinct parity
classes, hence at least `2^d` ticks — regardless of geometry, mesh, or lattice.
-/

import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Tactic

open Function

namespace MetaPrinciple
namespace LatticeFree

/-- The space of `d` binary parities is the funspace `Fin d → Bool`. -/
@[simp] def Pattern (d : Nat) := (Fin d → Bool)

instance (d : Nat) : Fintype (Pattern d) := inferInstance

/-- Cardinality: `|Pattern d| = 2^d`. -/
lemma card_pattern (d : Nat) :
  Fintype.card (Pattern d) = 2 ^ d := by
  classical
  -- `card (ι → α) = card α ^ card ι`
  simpa [Pattern] using (Fintype.card_fun : Fintype.card (Fin d → Bool) = _)

/-- There is no surjection from a set of size `T` onto `2^d` patterns when `T < 2^d`. -/
lemma no_surj_small (T d : Nat) (hT : T < 2 ^ d) :
  ¬ ∃ f : Fin T → Pattern d, Surjective f := by
  classical
  intro h
  rcases h with ⟨f, hf⟩
  -- A surjection admits a right inverse `g`
  obtain ⟨g, hg⟩ := hf.hasRightInverse
  have hginj : Injective g := by
    intro y₁ y₂ hgy
    -- apply `f` to both sides and use `hg : RightInverse g f`
    have : f (g y₁) = f (g y₂) := congrArg f hgy
    simpa [RightInverse, hg] using this
  -- Hence `card (Pattern d) ≤ card (Fin T) = T`
  have hcard : Fintype.card (Pattern d) ≤ Fintype.card (Fin T) :=
    Fintype.card_le_of_injective _ hginj
  have : 2 ^ d ≤ T := by
    simpa [Fintype.card_fin, card_pattern d] using hcard
  exact (lt_of_le_of_lt this hT).false

/-- Any pass (finite sequence) that covers all parity patterns has length ≥ `2^d`. -/
lemma min_ticks_cover {d T : Nat} (pass : Fin T → Pattern d)
  (covers : Surjective pass) : 2 ^ d ≤ T := by
  classical
  -- direct from `no_surj_small`
  by_contra h
  exact (no_surj_small T d (lt_of_not_ge h)) ⟨pass, covers⟩

/-- Generalized “no cover with fewer than 2^d” statement over an arbitrary
state space via a parity map. This is the reusable variant you asked for. -/
lemma no_cover_with_period_lt_two_pow_d'
  {V : Type} (d T : Nat)
  (π : V → Pattern d) (hπ : Surjective π) :
  T < 2 ^ d → ¬ ∃ f : Fin T → V, Surjective f := by
  intro hT h
  rcases h with ⟨f, hf⟩
  -- compose and reduce to the core no-surjection lemma
  have hcov : Surjective (fun i : Fin T => π (f i)) := hπ.comp hf
  -- if there were a surjection, we contradict `no_surj_small`
  have : ¬ ∃ g : Fin T → Pattern d, Surjective g :=
    no_surj_small T d hT
  exact this ⟨(fun i => π (f i)), hcov⟩

/-- Lifting the bound through a parity map from arbitrary state space. -/
lemma min_ticks_cover_states {d T : Nat} {State : Type}
  (parity : State → Pattern d)
  (seq : Fin T → State)
  (covers : Surjective (fun i => parity (seq i))) : 2 ^ d ≤ T := by
  classical
  -- Compose and apply the previous lemma
  have : Surjective (fun i : Fin T => (parity (seq i))) := covers
  simpa using (min_ticks_cover (d := d) (T := T) (fun i => parity (seq i)) this)

/-- Specialization to `d = 3`: at least eight ticks are required to cover all parities. -/
lemma eight_tick_min {T : Nat} (pass : Fin T → Pattern 3)
  (covers : Surjective pass) : 8 ≤ T := by
  simpa using (min_ticks_cover (d := 3) (T := T) pass covers)

end LatticeFree
end MetaPrinciple
