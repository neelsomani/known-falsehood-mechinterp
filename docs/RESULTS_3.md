# Ablation studies

## Directional ablation results

The previous ablation experiments identified a direction that linearly separates declared-true and declared-false statements at the end of the statement span, but removing this direction did not reliably affect downstream consequence selection. Further analysis showed that perturbations at the statement token did not mediate into the final decision representation, and even aggressive multi-layer ablations at the decision token could be compensated by later computation. This suggests that the stance signal used for consequence selection may not align with a single global probe direction across prompt formats or token positions.

To directly test whether epistemic stance is causally represented at the site that drives the model’s decision, we adopt a pairwise activation-patching approach. For each underlying consequence question, we construct two prompts that are identical except for epistemic stance (declared-true vs. declared-false). We run both prompts and capture the hidden state at a chosen layer and at the final prompt token, which is the representation from which the A/B logits are computed. We then perform a counterfactual intervention by overwriting the hidden state from one stance condition with the hidden state from the other, while keeping the prompt text fixed. If the model’s behavior shifts toward the patched stance, this provides direct causal evidence that the internal representation at that site mediates how epistemic stance influences downstream reasoning.

This approach avoids assumptions about global linear directions or cross-prompt alignment and instead tests the full high-dimensional representation induced by stance under identical conditions. Because the intervention is localized to a specific layer and token position and swaps only the internal state associated with epistemic stance, it constitutes a strong mechanistic test of whether stance is a causal control variable for premise-conditioned inference in the model.

## Behavioral effects of delta-direction ablation

We evaluated the effect of delta-direction ablation on both truth judgments and premise-conditioned consequence reasoning. The intervention removed, at the final decision site (last layer, last prompt token), the projection of the hidden state along the per-pair stance difference
Δ = h_declared-true − h_declared-false, normalized and applied with α = 1.0.

### Truth task

Performance on the truth classification task was unchanged by the intervention.

Before ablation:

* True accuracy: 0.975 (195/200)
* False accuracy: 0.955 (191/200)
* Overall accuracy: 0.965 (386/400)
* Unknown rate: 0.00%

After ablation:

* True accuracy: 0.975 (195/200)
* False accuracy: 0.955 (191/200)
* Overall accuracy: 0.965 (386/400)
* Unknown rate: 0.00%

Thus, delta-direction ablation does not disrupt the model’s ability to judge factual truth or parse negation in isolation.

### Consequence task

Performance on the premise-conditioned consequence task showed no meaningful degradation under ablation.

Before ablation:

* Overall accuracy: 0.666 (853/1280)
* Unparsed outputs: 0.00%

After ablation:

* Overall accuracy: 0.662 (848/1280)
* Unparsed outputs: 0.00%

The difference corresponds to 5 examples out of 1280 and is within noise.

Breakdowns by stance, proposition type, and template family likewise showed only minor fluctuations. No condition exhibited a systematic accuracy collapse following ablation.

### Summary

Delta-direction ablation at the final decision site leaves:

* truth judgment accuracy unchanged
* consequence-selection accuracy unchanged within noise

This holds despite the fact that the same intervention produces large and systematic shifts in answer logits and margins.

## Local delta-direction ablation across all layers

For the layer-sweep experiments, we subsample the set of stance pairs to 150 examples rather than using the full dataset (≈600 pairs). Computing per-pair local deltas requires two forward passes per example per layer, making a full sweep prohibitively expensive (on the order of tens of hours). Because the goal of this sweep is to identify whether there exists a layer (or contiguous band of layers) at which removing the stance-conditioned internal difference measurably degrades premise-conditioned reasoning, a smaller but representative subset is sufficient. The intervention is strong and example-specific, so any genuine causal bottleneck should manifest as a clear accuracy drop even at this reduced scale. Once candidate layers are identified, we can rerun targeted experiments at those layers using the full dataset for confirmation.

The layerwise results for a subset of the evaluation (300 rows) are included. The stance-conditioning bottleneck lives in layers ~30–45, with maximal causal impact around layer ~40. The local deltas are generally in the same direction:

{
  "n": 150,
  "cos_to_mean": {
    "mean": 0.4237349331378937,
    "std": 0.2707286775112152,
    "p05": -0.10179489105939865,
    "p50": 0.4580126404762268,
    "p95": 0.780331015586853
  },
  "abs_cos_to_mean": {
    "mean": 0.44999226927757263,
    "std": 0.2243930697441101,
    "p05": 0.08030397444963455,
    "p50": 0.4580126404762268,
    "p95": 0.780331015586853
  },
  "pairwise_cos": {
    "mean": 0.1788312941789627,
    "std": 0.34985262155532837,
    "p05": -0.3784124255180359,
    "p50": 0.15232263505458832,
    "p95": 0.760366678237915
  },
  "pca_var_explained_top10": [
    0.39093324542045593,
    0.23926103115081787,
    0.09603725373744965,
    0.0657278448343277,
    0.06230822578072548,
    0.045001525431871414,
    0.03697717934846878,
    0.024565577507019043,
    0.023927519097924232,
    0.015260549262166023
  ]
}
