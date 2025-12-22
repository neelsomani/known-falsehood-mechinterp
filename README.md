# Do Models Encode Epistemic Stance in Reasoning?

We investigate whether language models distinguish between a proposition's content and its conditional licensing as a premise under an explicit assumption. Specifically, we ask whether internal activations differ when the same proposition X is used as a premise for reasoning under different assumptions about X (e.g., explicitly declared true versus explicitly declared false), even when the surface content of X is held fixed.

For example, consider the proposition X = "Paris is the capital of France." When presented with "It is true that: Paris is the capital of France," a model should reason from X and endorse consequences consistent with it. When presented with "It is false that: Paris is the capital of France," the model should instead reason from ¬X when reasoning under the stated assumption, despite the propositional content being identical in both cases. Importantly, the model is not asked to judge whether X is true in the real world, but to condition its downstream reasoning on how X is framed as an assumed premise. Correct behavior therefore requires tracking not just what X says, but how it is licensed for use as a premise.

We construct a controlled dataset of short factual statements in four classes (declared-true vs. declared-false, X_true vs. X_false) using minimal lexical edits and diverse stance templates to reduce format artifacts. We record residual stream activations at multiple token positions and layers, and train linear probes to discriminate declared-true versus declared-false conditions while holding propositional content fixed. To test whether any separable representation is causally involved in reasoning, we ablate the inferred "stance direction" (and/or top-weight neurons) and measure changes in downstream, premise-conditioned behavior: forced-choice consequence selection, consistency across multi-step inference, and whether the model propagates X versus ¬X when X is explicitly marked as false.

Our results provide evidence that models encode epistemic stance as a separable, assumption-scoped signal that is causally implicated in premise-conditioned inference. Naive ablations of a single global probe direction have limited behavioral effect, but delta-based interventions aligned to the decision site reveal a low-dimensional, shared stance subspace in mid-to-late layers. Removing this subspace substantially degrades the model's ability to respect declared-false premises in downstream consequence selection.

## Setup

Install dependencies (Python 3.9+):
```bash
pip install "torch>=2.1" "transformers>=4.41" accelerate
```
*Optional*: `pip install hf-transfer` for faster local cache use.

## Run behavioral eval

1) Build the derived dataset files from the wide facts CSV:
   ```bash
   python scripts/convert_wide_to_long.py --input dataset/facts.csv
   ```
   This regenerates `dataset/data.csv`, `dataset/consequences.csv`, and `dataset/splits.json`.

2) Run the eval:
   ```bash
   python scripts/run_behavioral_eval.py --model meta-llama/Llama-3.1-70B-Instruct
   ```
   Use `--limit-truth-per-class` or `--limit-consequence` to sample fewer items, and `--show-prompts` to print prompts and outputs as they run.
   `scripts/build_train_consequence_questions.py` and `scripts/llama_interactive.py` are a debugging utilities and do not need to be run.

The evaluation shows that the model can distinguish between raw, declared-true, and declared-false.

## Capture activations

Capture residual stream activations at the entity/object token and final token:
   ```bash
   python scripts/capture_activations.py --model meta-llama/Llama-3.1-70B-Instruct
   ```
   This writes `dataset/activations_train.pt` (or `activations_{split}.pt` for other splits). Use `--split train` or `--split test` as needed.

Train linear probes on residual stream activations and evaluate on the held-out split.

* **Task A (core stance; primary claim):**
declared-true(X_true) vs declared-false(X_true)
→ tests pure epistemic stance

* Task B (generalization):
declared-true(X_false) vs declared-false(X_false)
→ tests whether the same stance representation applies when the proposition contradicts world knowledge

* Task C (control):
X_false vs declared-false(X_false)
→ wrapper detection only

This runs logistic regression probes per layer and per position (entity vs final token) for Tasks A/B/C, and reports mean AUROC across 3 seeds.

```bash
python scripts/run_probes.py
```

Outputs `dataset/probe_aurocs.csv` with AUROC by task, layer, and position. Note: the bare condition (T_BARE) is included in both train and test so Task C (bare vs declared-false on X_false) is evaluable. Template disjointness applies to the paired TRUE/FALSE templates.

The output shows that declared-true is linearly separable from declared-false in the activations.

## Delta-direction ablation

Build prompts with statement-final token indices:
```bash
python scripts/build_intervention_prompts.py
```

Compute per-pair stance deltas at the decision site and remove the projection along that delta:
```bash
python scripts/run_intervention_delta_ablation.py \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --layer last \
  --delta-mode per-pair \
  --direction both \
  --alpha 1.0 \
  --max-pairs 100
```

Outputs `dataset/intervention_delta_ablation.jsonl` with per-pair margins and predictions. This produces large, systematic logit-margin shifts at the output, confirming that stance-conditioned differences are present at the decision state. However, last-layer ablation alone does not necessarily induce large accuracy drops, motivating a layerwise search for where stance becomes behaviorally binding.

## Evaluate local delta direction ablation

The failure of single-direction ablations is itself evidence against a trivial control feature. It motivates a more local, decision-aligned causal analysis.

First, sweep layers with per-example (local) deltas to identify where ablation has the biggest impact on consequence accuracy:

```bash
python scripts/run_layerwise_delta_eval.py \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --layers all \
  --max-pairs 150
```

We subsample stance pairs during the sweep to make full-layer evaluation tractable. The goal is to localize the causal layer band, after which selected layers can be rerun with more pairs. This writes per-layer deltas to `dataset/delta_directions/` and per-layer eval results to `dataset/layerwise_delta_eval.jsonl`. In our runs, the largest drop appeared around layer 40.

Next, analyze delta alignment at that layer:

```bash
python scripts/analyze_delta_alignment.py --delta-pt dataset/delta_directions/delta_layer_040.pt
```

Observe that most of the local deltas are in the same direction, with mean cosine to the mean direction ≈ 0.42 and top-2 PCs explaining ≈ 63% of variance (example run with 150 pairs). Finally, compare baseline vs local deltas vs global directions (mean, random, and PCA subspaces). Use uncentered SVD for the PCA subspaces:

```bash
python scripts/run_avg_delta_eval.py \
  --delta-pt delta_layer_040.pt \
  --layer 40 \
  --pca-ks 1,2,3 \
  --pca-mode uncentered
```

This writes a JSON summary to `dataset/avg_delta_eval.json`. At layer 40, ablating a 2-3 dimensional subspace estimated from aligned stance deltas substantially reduces consequence accuracy (approaching the effect of per-example local delta ablation), whereas a single global direction (mean or top-1 component) has a much weaker effect. Random directions do not have any impact.
