## Local delta-direction ablation across all layers

For the layer-sweep experiments, we subsample the set of stance pairs to 150 examples rather than using the full dataset (≈600 pairs). Computing per-pair local deltas requires two forward passes per example per layer, making a full sweep prohibitively expensive (on the order of tens of hours). Because the goal of this sweep is to identify whether there exists a layer (or contiguous band of layers) at which removing the stance-conditioned internal difference measurably degrades premise-conditioned reasoning, a smaller but representative subset is sufficient. The intervention is strong and example-specific, so any genuine causal bottleneck should manifest as a clear accuracy drop even at this reduced scale. Once candidate layers are identified, we can rerun targeted experiments at those layers using the full dataset for confirmation.

The layerwise results for a subset of the evaluation (300 rows) are included. The stance-conditioning bottleneck lives in layers ~30–45, with maximal causal impact around layer ~40.

## Mean of local delta-direction ablation

The local deltas are generally in the same direction:

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

That suggests subtracting the mean:

{
  "layer": 40,
  "alpha": 1.0,
  "delta_pt": "dataset/delta_directions/delta_layer_040.pt",
  "results": {
    ...,
    "local_delta": {
      "acc": 0.5366666666666666,
      "unparsed_pct": 0.0,
      "skipped_missing_delta": 980,
      "total": 300
    },
    "mean_delta": {
      "acc": 0.6533333333333333,
      "unparsed_pct": 0.0,
      "skipped_missing_delta": 980,
      "total": 300
    },
    "random_delta": {
      "acc": 0.6766666666666666,
      "unparsed_pct": 0.0,
      "skipped_missing_delta": 980,
      "total": 300
    }
  }
}

Clearly it's not a 1D representation, though the 1D projection has some effect.

## PCA subspace ablations: stance is low-D but not 1-D

Using PCA-based subspaces at layer 40:

{
  "layer": 40,
  "alpha": 1.0,
  "delta_pt": "dataset/delta_directions/delta_layer_040.pt",
  "results": {
    "baseline": {
      "acc": 0.67,
      "unparsed_pct": 0.0,
      "skipped_missing_delta": 980,
      "total": 300
    },
    "local_delta": {
      "acc": 0.5366666666666666,
      "unparsed_pct": 0.0,
      "skipped_missing_delta": 980,
      "total": 300
    },
    "mean_delta": {
      "acc": 0.65,
      "unparsed_pct": 0.0,
      "skipped_missing_delta": 980,
      "total": 300
    },
    "random_delta": {
      "acc": 0.68,
      "unparsed_pct": 0.0,
      "skipped_missing_delta": 980,
      "total": 300
    },
    "pca_k1": {
      "acc": 0.6766666666666666,
      "unparsed_pct": 0.0,
      "skipped_missing_delta": 980,
      "total": 300
    },
    "pca_k2": {
      "acc": 0.5966666666666667,
      "unparsed_pct": 0.0,
      "skipped_missing_delta": 980,
      "total": 300
    },
    "pca_k3": {
      "acc": 0.6633333333333333,
      "unparsed_pct": 0.0,
      "skipped_missing_delta": 980,
      "total": 300
    }
  }
}

Key insight:

* Removing *more* variance does **not** monotonically reduce accuracy.
* PCA components are ordered by variance, not causal relevance.
* Some higher-variance components are not stance-critical and may even support correct answers.

Conclusion: The causal stance signal lives in a **small subspace (≈2D)**, but not necessarily aligned with PCA variance ordering.

## Final finding: Uncentered PCA explains it

When we estimate a stance subspace using PCA on mean-centered stance deltas, the resulting components capture only how stance varies across examples, not the global stance shift itself. Ablating the projection of the decision-state hidden vector onto this centered subspace degrades performance but does not fully collapse premise-conditioned reasoning, indicating that removing structured variation alone is insufficient. Empirically, we find that the mean of the stance deltas constitutes an essential part of the causal signal: subtracting only the mean produces a weak effect, subtracting only centered variation produces a stronger but incomplete effect, and subtracting both together produces the largest degradation. This motivates using an uncentered delta subspace, estimated via SVD on the raw (sign-aligned) deltas, which spans both the mean stance direction and its dominant modes of variation.

{
  "layer": 40,
  "alpha": 1.0,
  "delta_pt": "dataset/delta_directions/delta_layer_040.pt",
  "results": {
    "baseline": {
      "acc": 0.67,
      "unparsed_pct": 0.0,
      "skipped_missing_delta": 980,
      "total": 300
    },
    "local_delta": {
      "acc": 0.5366666666666666,
      "unparsed_pct": 0.0,
      "skipped_missing_delta": 980,
      "total": 300
    },
    "mean_delta": {
      "acc": 0.65,
      "unparsed_pct": 0.0,
      "skipped_missing_delta": 980,
      "total": 300
    },
    "random_delta": {
      "acc": 0.68,
      "unparsed_pct": 0.0,
      "skipped_missing_delta": 980,
      "total": 300
    },
    "pca_k1": {
      "acc": 0.6633333333333333,
      "unparsed_pct": 0.0,
      "skipped_missing_delta": 980,
      "total": 300
    },
    "pca_k2": {
      "acc": 0.5733333333333334,
      "unparsed_pct": 0.0,
      "skipped_missing_delta": 980,
      "total": 300
    },
    "pca_k3": {
      "acc": 0.56,
      "unparsed_pct": 0.0,
      "skipped_missing_delta": 980,
      "total": 300
    }
  }
}

Projecting the hidden state onto this uncentered stance subspace and subtracting that projection most closely reproduces the effect of per-example local delta ablations, indicating that epistemic stance is implemented as a low-dimensional span consisting of a global bias plus structured contextual modulation, rather than purely as variance around a mean or as a single direction.
