# Interpreting the Probing Results  
## Epistemic Stance Encoding in Llama-3.1-70B

This section interprets the results from `probe_aurocs.csv` produced in **Step 4: Probing**, with the goal of determining whether the model encodes *epistemic stance* (assumed true vs. assumed false) as a separable internal signal, distinct from propositional content.

---

## Recap: What Each Task Tests

### Task A — Core Stance (Primary Claim)
**declared-true(X_true) vs. declared-false(X_true)**  
- Proposition content is fixed and factually true.
- Tests whether the model internally distinguishes *assumed true* vs. *assumed false* purely via epistemic stance.

### Task B — Generalization
**declared-true(X_false) vs. declared-false(X_false)**  
- Proposition content is fixed but factually false.
- Tests whether the same stance representation generalizes when world knowledge disagrees.

### Task C — Control (Wrapper Detection)
**X_false (bare) vs. declared-false(X_false)**  
- Tests whether probes are detecting superficial wrapper / format cues rather than stance.

---

## Summary of Key Findings

### 1. Strong Stance Signal at the Final Token (Tasks A & B)

- For **both Task A and Task B**, linear probes achieve:
  - Near-chance AUROC (~0.5) in **layers 0–1**
  - Rapid rise by **layer 2–3**
  - **Near-perfect AUROC (≈0.99–1.00)** from early-mid layers onward
- This pattern holds **only at the final token position**.

**Interpretation:**  
The model encodes epistemic stance as a clean, linearly separable feature that is integrated into the *sentence-level representation* by early layers and preserved through the network.

---

### 2. No Stance Signal at the Entity / Content Token (Tasks A & B)

- At the **entity/content token position**, AUROC stays ~0.5 across all layers.
- No linear probe can distinguish declared-true vs. declared-false using the content token alone.

**Interpretation:**  
Propositional *content* and epistemic *licensing* are represented separately.  
The content token embedding itself does not collapse or flip under different assumed stances.

This strongly supports your core framing:
> the model tracks *how* a proposition is licensed independently of *what* the proposition says.

---

### 3. Task B Tracks Task A (Generalization Success)

- Task B (false propositions) shows essentially the **same AUROC–by-layer profile** as Task A.
- High separability appears at the same layers and positions.

**Interpretation:**  
The stance representation is **not tied to factual correctness** or “truthiness heuristics.”
Instead, it reflects an assumption-scoped signal:  
> “In this context, treat X as true / false,” regardless of world knowledge.

---

### 4. Task C Confirms Wrapper Detectability (Control Works)

- Task C shows **near-perfect AUROC from layer 0** at *both* positions.
- This is expected: bare vs. wrapped inputs differ lexically and structurally.

Notably:
- There is a **mid-layer dip** in AUROC at the final token (≈0.7–0.8),
- followed by recovery to ≈1.0 in later layers.

**Interpretation:**  
- Early layers encode obvious lexical format cues.
- Mid layers may transiently abstract away from surface form.
- Final layers again encode a clean distinction (likely semantic: “this statement is negated”).

Crucially, this Task C behavior is **qualitatively different** from Tasks A/B:
- In A/B, the signal *emerges later* and *only at the final token*.
- In C, the signal is trivial and omnipresent.

This separation rules out the explanation that Tasks A/B are driven by simple wrapper detection.

---

## Layer × Position Interpretation

| Position | Task A/B | Task C |
|--------|----------|--------|
| Entity token | No separable signal | Strong signal |
| Final token | Strong stance signal | Strong signal |

**Key takeaway:**  
Epistemic stance is **not stored locally in content tokens**, but is integrated into a **global sentence-level representation**.

This aligns with the idea that the model maintains:
- a stable content representation, and
- a separate control-like signal governing how that content is used in reasoning.

---

## Probe Site Selection

For causal interventions, we select a single (layer, position) at which to extract a stance direction. To minimize contamination from superficial wrapper cues, we choose the site that maximizes

AUROC(Task A) − AUROC(Task C)

on the held-out split, restricted to content positions. This criterion prioritizes locations where epistemic stance is strongly decodable while wrapper-only signals are weakest. In our experiments, this corresponds to the final-token residual stream at layer 15.

As a robustness check, we also consider the earliest layer at which Task A reaches its maximum AUROC (layer 3, final token), and verify that qualitative intervention effects are consistent across both sites.

---

## What We Can Claim

Based on these results alone:

- The model **encodes epistemic stance as a linearly separable internal signal**.
- This signal:
  - is distinct from propositional content,
  - generalizes across true and false propositions,
  - is not reducible to wrapper-format cues,
  - emerges early and persists through late layers.
- The encoding is **assumption-scoped**, not a global belief update.

---

## What Comes Next:

### 1. Choose the Intervention Site
- Best candidate:
  - **Final token**
  - Earliest layer where AUROC saturates (often layer 3–6)
- Extract normalized probe direction `w`.

### 2. Perform Direction Ablation
Apply:
a' = a − α (a · w) w

Test effects on:
- declared-false consequence selection,
- multi-step reasoning consistency.

### 3. Critical Causal Test
If ablating `w` selectively breaks:
- reasoning under declared-false assumptions,
- while leaving surface fluency intact,

then you have strong causal evidence that:
> the stance direction *modulates inference*, not just classification.
