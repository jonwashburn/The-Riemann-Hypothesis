import Mathlib.Data.Fintype.Card
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Bool.Basic
import Mathlib.Tactic

set_option autoImplicit true
open Function

/-- Our concrete model of vertices: 3 parity bits. -/
abbrev V_real := Fin 3 → Bool

/-- Coordinate parity (just projection). -/
def φ (i : Fin 3) (v : V_real) : Bool := v i

/-- `parity_complete` is trivial: for any `i,v` just take `b := v i`. -/
lemma parity_complete (i : Fin 3) (v : V_real) : ∃ b, φ i v = b := ⟨v i, rfl⟩

/-- `parity_ext` is just function extensionality. -/
lemma parity_ext {v w : V_real}
  (h : ∀ i : Fin 3, φ i v = φ i w) : v = w := by
  funext i; exact h i

variable {V : Type*} [Fintype V]
variable (par : Fin 3 → V → Bool)

/-- Structure map from the concrete model onto `V`. -/
variable (π : V_real → V)

/-- π preserves parities (your `hπ`). -/
variable (hπ : ∀ v i, par i (π v) = φ i v)

/-- Parities separate points in V (your `sep`). -/
variable (sep : ∀ ⦃v w : V⦄, (∀ i, par i v = par i w) → v = w)

/-- If you’ve already proved this elsewhere, you can use it directly. -/
variable (π_surj : Surjective π)

/-- Embed `V` into the concrete cube by reading off its 3 parities. -/
def κ : V → V_real := fun v i => par i v

/-- `sep` ⇒ κ is injective. -/
lemma κ_injective : Injective (κ par) := by
  intro v w h
  have hw : ∀ i, (par i v) = (par i w) := by
    intro i; simpa [κ, par] using congrArg (fun f => f i) h
  exact sep hw

/-- The cheap cardinality proof: `|V| = |V_real| = 8`. -/
theorem card_V_exactly_8
  [DecidableEq V] :
  Fintype.card V = 8 := by
  classical
  have h₂ : Fintype.card V_real ≤ Fintype.card V :=
    Fintype.card_le_of_surjective π π_surj
  have h₁ : Fintype.card V ≤ Fintype.card V_real :=
    Fintype.card_le_of_injective (κ par) (κ_injective (par := par) (sep := sep))
  have card_Vreal_2pow3 :
      Fintype.card V_real = 2 ^ Fintype.card (Fin 3) := by
    simpa [V_real] using Fintype.card_fun (Fin 3) Bool
  have : Fintype.card (Fin 3) = 3 := by simpa using Fintype.card_fin 3
  have card_Vreal :
      Fintype.card V_real = 2 ^ 3 := by simpa [this] using card_Vreal_2pow3
  have : Fintype.card V = Fintype.card V_real := le_antisymm h₁ h₂
  have : Fintype.card V = 2 ^ 3 := by simpa [card_Vreal] using this
  simpa using (by decide : 2 ^ 3 = (8 : Nat))
