# Do Models Encode Epistemic Stance in Reasoning?

We investigate whether language models distinguish between a proposition's content and its conditional licensing as a premise under an explicit assumption. Specifically, we ask whether internal activations differ when the same proposition X is used as a premise for reasoning under different assumptions about X (e.g., explicitly declared true versus explicitly declared false), even when the surface content of X is held fixed.

For example, consider the proposition X = "Paris is the capital of France." When presented with "It is true that: Paris is the capital of France," a model should reason from X and endorse consequences consistent with it. When presented with "It is false that: Paris is the capital of France," the model should instead reason from ¬X when reasoning under the stated assumption, despite the propositional content being identical in both cases. Importantly, the model is not asked to judge whether X is true in the real world, but to condition its downstream reasoning on how X is framed as an assumed premise. Correct behavior therefore requires tracking not just what X says, but how it is licensed for use as a premise.

We construct a controlled dataset of short factual statements in four classes (declared-true vs. declared-false, X_true vs. X_false) using minimal lexical edits and diverse stance templates to reduce format artifacts. We record residual stream activations at multiple token positions and layers, and train linear probes to discriminate declared-true versus declared-false conditions while holding propositional content fixed. To test whether any separable representation is causally involved in reasoning, we ablate the inferred "stance direction" (and/or top-weight neurons) and measure changes in downstream, premise-conditioned behavior: forced-choice consequence selection, consistency across multi-step inference, and whether the model propagates X versus ¬X when X is explicitly marked as false.

Our results provide evidence that language models encode epistemic stance as a separable, assumption-scoped internal signal that modulates how propositional content is used in downstream inference, and that manipulating this signal selectively disrupts the model's ability to treat explicit negation as a constraint on reasoning under assumed premises.

## Quickstart: interactive Llama 70B chat

1) Install dependencies (Python 3.9+):
   ```bash
   pip install "torch>=2.1" "transformers>=4.41" accelerate
   ```
   *Optional*: `pip install hf-transfer` for faster local cache use.

2) Run the interactive loop (uses the chat template described in `docs/AGENTS.md`):
   ```bash
   python scripts/llama_interactive.py --model meta-llama/Llama-3.1-70B-Instruct
   ```
   Type at the `You>` prompt; exit with `exit`/`quit`/Ctrl+C/Ctrl+D. Adjust `--temperature 0` for greedy generation or tweak `--max-new-tokens`, `--top-p`, and `--dtype` as needed.

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
   `scripts/build_train_consequence_questions.py` and `scripts/compute_entity_token_indices.py` are debugging utilities and do not need to be run.

## Capture activations

Capture residual stream activations at the entity/object token and final token:
   ```bash
   python scripts/capture_activations.py --model meta-llama/Llama-3.1-70B-Instruct --split train
   ```
   This writes `dataset/activations_train.pt` (or `activations_{split}.pt` for other splits). Use `--split test` or `--split all` as needed.

## Run probes

Train linear probes on residual stream activations and evaluate on the held-out split.
This runs logistic regression probes per layer and per position (entity vs final token) for Tasks A/B/C from `docs/AGENTS.md`,
and reports mean AUROC across 3 seeds.

```bash
python scripts/run_probes.py
```

Outputs `dataset/probe_aurocs.csv` with AUROC by task, layer, and position.
