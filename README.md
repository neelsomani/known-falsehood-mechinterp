# Do Models Encode Epistemic Stance in Reasoning?

We investigate whether language models distinguish between a proposition's content and its conditional licensing as a premise under an explicit assumption. Specifically, we ask whether internal activations differ when the same proposition X is used as a premise for reasoning under different assumptions about X (e.g., explicitly declared true versus explicitly declared false), even when the surface content of X is held fixed.

For example, consider the proposition X = "Paris is the capital of France." When presented with "It is true that: Paris is the capital of France," a model should reason from X and endorse consequences consistent with it. When presented with "It is false that: Paris is the capital of France," the model should instead reason from ¬X when reasoning under the stated assumption, despite the propositional content being identical in both cases. Importantly, the model is not asked to judge whether X is true in the real world, but to condition its downstream reasoning on how X is framed as an assumed premise. Correct behavior therefore requires tracking not just what X says, but how it is licensed for use as a premise.

We construct a controlled dataset of short factual statements in four classes (declared-true vs. declared-false, X_true vs. X_false) using minimal lexical edits and diverse stance templates to reduce format artifacts. We record residual stream activations at multiple token positions and layers, and train linear probes to discriminate declared-true versus declared-false conditions while holding propositional content fixed. To test whether any separable representation is causally involved in reasoning, we ablate the inferred "stance direction" (and/or top-weight neurons) and measure changes in downstream, premise-conditioned behavior: forced-choice consequence selection, consistency across multi-step inference, and whether the model propagates X versus ¬X when X is explicitly marked as false.

Our results provide evidence that models encode epistemic stance as a separable, assumption-scoped signal that is causally implicated in premise-conditioned inference. Naive ablations of a single global probe direction have limited behavioral effect, but delta-based interventions aligned to the decision site reveal a low-dimensional, shared stance subspace in mid-to-late layers. Removing this subspace substantially degrades the model's ability to respect declared-false premises in downstream consequence selection.

## Goal

The goal of this work is to determine whether epistemic stance—whether a proposition is taken as assumed true or assumed false—is represented internally as a distinct, assumption-scoped control signal, and whether that signal causally mediates downstream reasoning rather than merely reflecting surface negation or world-knowledge heuristics.

## Setup

Install dependencies (Python 3.9+):
```bash
pip install "torch>=2.1" "transformers>=4.41" accelerate
```
Optional: `pip install hf-transfer` for faster local cache use.

## Controlled setup to isolate epistemic stance

We construct a dataset of short factual statements X and minimally edited variants ¬X, paired with consequence questions that require propagating the assumed premise.

Each statement appears under multiple epistemic conditions:

* declared-true(X)
* declared-false(X)
* declared-true(¬X)
* declared-false(¬X)
* bare(¬X) (control)

Critically:

* Surface content is held fixed within each comparison.
* World-knowledge correctness is orthogonalized from epistemic stance.
* Consequence questions are designed so that correct answers depend on how the premise is licensed, not on real-world truth.

This lets us ask whether the model tracks assumed truth value independently of content and knowledge.

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

The evaluation shows that the model can distinguish between raw, declared-true, and declared-false. The evaluation will be run again after the intervention.

## Probing for separable representations

We first ask a representational question: Does the model encode epistemic stance as a linearly separable internal signal?

We record residual stream activations at multiple layers and token positions and train linear probes for three tasks:

* Task A: declared-true(X_true) vs declared-false(X_true)
* Task B: declared-true(X_false) vs declared-false(X_false)
* Task C: bare(¬X) vs declared-false(¬X) (wrapper control)

1. Capture residual stream activations at the entity/object token and final token:
   ```bash
   python scripts/capture_activations.py --model meta-llama/Llama-3.1-70B-Instruct
   ```
   This writes `dataset/activations_train.pt` (or `activations_{split}.pt` for other splits). Use `--split train` or `--split test` as needed.

2. Run the classifier:
   ```bash
   python scripts/run_probes.py
   ```
   Outputs `dataset/probe_aurocs.csv` with AUROC by task, layer, and position. Note: the bare condition (T_BARE) is included in both train and test so Task C (bare vs declared-false on X_false) is evaluable. Template disjointness applies to the paired TRUE/FALSE templates.

### Findings

* For Tasks A and B, epistemic stance becomes near-perfectly linearly separable in early-mid layers, but only at the final token of the statement.
* No stance signal is detectable at the content/entity token.
* Task B mirrors Task A, showing the signal is independent of world knowledge.
* Task C behaves qualitatively differently, confirming that Tasks A/B are not driven by trivial wrapper cues.

### Interpretation

The model maintains:

* a stable content representation, and
* a separate, sentence-level control signal indicating how that content is licensed as a premise.

This establishes representational separability (but not causality).

## Naive causal test fails

We next ask the causal question: Does removing the linearly separable stance direction disrupt premise-conditioned reasoning?

We ablate a global probe direction at the statement-final token and measure downstream consequence accuracy.

### Result

* Logit margins shift substantially.
* But consequence accuracy remains essentially unchanged.

### Interpretation

This negative result is informative:

* Epistemic stance is not implemented as a single global direction.
* Removing a direction at the statement token does not guarantee intervention at the decision site.
* The model can reconstruct or compensate for such perturbations downstream.

This rules out a trivial "one-direction control knob" model and motivates a more localized causal analysis.

## Decision-site, pairwise intervention (true causal test)

To directly test causality, we move the intervention to the decision site and avoid assumptions about global alignment.

For each consequence question, we construct two prompts that differ only in epistemic stance and capture the hidden state at the final decision token.

We define the local stance delta:

$$ \delta_i = h_\text{declared-true} − h_\text{declared-false} $$

and perform counterfactual interventions by removing the projection along \delta_i at a chosen layer.

This tests whether the internal difference induced by stance actually mediates the model's decision.

1. Build prompts with statement-final token indices:
   ```bash
   python scripts/build_intervention_prompts.py --split train
   # Optionally produce the test split
   ```

This produces the prompts and pairs for the ablation test.

2. Compute per-pair stance deltas at the decision site and remove the projection along that delta:
   ```bash
   python scripts/run_intervention_delta_ablation.py \
     --model meta-llama/Llama-3.1-70B-Instruct \
     --layer last \
     --delta-mode per-pair \
     --direction both \
     --alpha 1.0 \
     --max-pairs 100
   ```
   Outputs `dataset/intervention_delta_ablation.jsonl` with per-pair margins and predictions.

This produces large, systematic logit-margin shifts at the output, confirming that stance-conditioned differences are present at the decision state. However, last-layer ablation alone does not necessarily induce large accuracy drops, motivating a layerwise search for where stance becomes behaviorally binding.

## Layer sweep reveals a causal bottleneck

We sweep this local delta ablation across layers. First, sweep layers with per-example (local) deltas to identify where ablation has the biggest impact on consequence accuracy:

```bash
python scripts/run_layerwise_delta_eval.py \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --layers all \
  --max-pairs 150
```

We subsample stance pairs during the sweep to make full-layer evaluation tractable. The goal is to localize the causal layer band, after which selected layers can be rerun with more pairs. This writes per-layer deltas to `dataset/delta_directions/` and per-layer eval results to `dataset/layerwise_delta_eval.jsonl`. In our runs, the largest drop appeared around layer 40.

### Findings

* Early layers: little effect.
* Mid-to-late layers (~30–45): large, systematic drops in consequence accuracy.
* Peak effect around layer ~40.

### Interpretation

Epistemic stance becomes behaviorally binding only once representations have been transformed into a decision-relevant form. This localizes a causal bottleneck in the network.

## Structure of the stance representation

Next, analyze delta alignment at that layer:

```bash
python scripts/analyze_delta_alignment.py --delta-pt dataset/delta_directions/delta_layer_040.pt
```

The cosine similarity of the delta directions motivates trying mean and PCA subspace ablations. Finally, compare baseline vs local deltas vs global directions (mean, random, and PCA subspaces). Use uncentered SVD for the PCA subspaces:

```bash
python scripts/run_avg_delta_eval.py \
  --delta-pt delta_layer_040.pt \
  --layer 40 \
  --pca-ks 1,2,3 \
  --pca-mode uncentered
```

This writes a JSON summary to `dataset/avg_delta_eval.json`. At layer 40, ablating a 2-3 dimensional subspace estimated from aligned stance deltas substantially reduces consequence accuracy (approaching the effect of per-example local delta ablation), whereas a single global direction (mean or top-1 component) has a much weaker effect. Random directions do not have any impact.

Findings:

* Local deltas are partially aligned but not identical.
* A single mean direction has weak effect.
* PCA reveals that stance lives in a small subspace (~2–3 dimensions).
* Variance-ordered PCA components are not causally ordered.

Crucially:

* Removing only centered variation is insufficient.
* Removing only the mean is insufficient.
* Removing an uncentered low-rank subspace (mean + structured variation) best reproduces the effect of per-example local ablations.

### Interpretation

Epistemic stance is implemented as:

* a low-dimensional control subspace,
* consisting of a global bias plus context-dependent modulation,
* not a single neuron or direction.

## Selectivity of the intervention

Finally, we re-run the truth evaluation control task:

```bash
python scripts/run_truth_subspace_eval.py
   --model meta-llama/Llama-3.1-70B-Instruct
   --delta-pt dataset/delta_directions/delta_layer_040.pt
   --layer 40
   --k 3
   --alpha 1.0
   --dtype bfloat16
   --limit-truth-per-class 200
   --seed 0
```

For the chosen ablation that subtracts the projection against the PCA-3 delta direction subspace:

* Output formatting remains intact (A/B and True/False parsing rates unchanged).
* Truth judgments are unchanged under the same global stance-subspace ablation.
* Premise-conditioned consequence accuracy degrades substantially under this ablation.

This demonstrates functional specificity within the evaluated regime: the intervention disrupts how premises are used for inference, rather than impairing propositional understanding, truth evaluation, or instruction following in these tasks.

## Final conclusion

Large language models encode epistemic stance as a distinct, low-dimensional, assumption-scoped control signal that is:

* separable from propositional content,
* independent of world knowledge,
* localized to a mid-to-late decision bottleneck,
* and causally responsible for premise-conditioned reasoning.

Naive ablations fail because stance is not globally aligned or one-dimensional. Only decision-site, structure-aware interventions reveal its causal role.

This provides direct mechanistic evidence that models internally track not just what a proposition says, but how it is licensed for inference.
