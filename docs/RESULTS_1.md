# Part 1: Behavioral Characterization (Preliminary Findings)

As an initial step, we characterize model behavior on two closely related tasks that separate factual truth evaluation from assumption-conditioned reasoning. On a Truth/False/Unknown task involving short geopolitical statements, the model achieves high accuracy (96.5% overall) with zero use of the “Unknown” option. This confirms that, under the current prompt format, the model continues to evaluate bare propositions against its internal world knowledge rather than defaulting to acceptance or assumption-taking. These results establish a stable behavioral baseline and verify that the task setup does not globally bias the model toward treating user-provided statements as true.

We then examine a consequence task that uses the same underlying propositions but embeds them in explicit stance framings (e.g., declared true vs. declared false) and asks the model to reason downstream under the stated assumption. Performance on this task is substantially lower (65.0% accuracy) and exhibits strong structure rather than random error. Accuracy varies systematically across stance polarity and template family. In particular, declared-false(X_true) cases are handled more reliably than declared-true(X_false) cases, suggesting that the model is more consistent at suppressing inferences from explicitly negated premises than at reasoning from counterfactual assumptions. Additionally, consequence accuracy differs sharply across surface realizations of epistemic stance, with some template families yielding near-80% accuracy and others approaching chance.

These preliminary behavioral results indicate a clear dissociation between factual truth evaluation and assumption-scoped reasoning. While the model robustly evaluates the truth of propositions in isolation, it only inconsistently treats stance-framed propositions as licensed premises for downstream inference, and this inconsistency is highly sensitive to linguistic realization. The structured nature of these failures suggests that any internal representation of epistemic stance, if present, is fragile and unevenly integrated with propositional content. This regime—where assumption-following is neither absent nor reliable—motivates the next stage of the project, which investigates whether such stance distinctions correspond to separable internal signals and whether those signals play a causal role in downstream reasoning.

Truth/False/Unknown task:
  True accuracy: 0.975 (195/200)
  False accuracy: 0.955 (191/200)
  Overall accuracy: 0.965 (386/400)
  Unknown percent: 0.00% (0/400)

Consequence task (train facts/templates):
  Accuracy: 0.666 (853/1280)
  Unparsed percent: 0.00% (0/1280)

Consequence breakdown by template family:
  declared_false(X_false) | family 1: acc 0.769 (123/160), unparsed 0.00% (0/160)
  declared_false(X_true) | family 1: acc 0.806 (129/160), unparsed 0.00% (0/160)
  declared_true(X_false) | family 1: acc 0.637 (102/160), unparsed 0.00% (0/160)
  declared_true(X_true) | family 1: acc 0.706 (113/160), unparsed 0.00% (0/160)
  declared_false(X_false) | family 2: acc 0.506 (81/160), unparsed 0.00% (0/160)
  declared_false(X_true) | family 2: acc 0.556 (89/160), unparsed 0.00% (0/160)
  declared_true(X_false) | family 2: acc 0.644 (103/160), unparsed 0.00% (0/160)
  declared_true(X_true) | family 2: acc 0.706 (113/160), unparsed 0.00% (0/160)

Final summary:
  Truth overall accuracy: 0.965
  Truth unknown percent: 0.00%
  Consequence accuracy: 0.666
  Consequence unparsed percent: 0.00%