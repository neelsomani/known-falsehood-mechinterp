#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def summarize(x: torch.Tensor) -> dict[str, float]:
    x = x.detach().cpu()
    q = torch.quantile(x, torch.tensor([0.05, 0.5, 0.95]))
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "p05": float(q[0].item()),
        "p50": float(q[1].item()),
        "p95": float(q[2].item()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta-pt", type=Path, required=True, help="Path to delta_layer_XXX.pt")
    ap.add_argument("--max-pairs", type=int, default=2000, help="Cap number of deltas to analyze")
    ap.add_argument(
        "--pairwise-samples",
        type=int,
        default=50000,
        help="Random pairwise cosine samples",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    payload = torch.load(args.delta_pt, map_location="cpu")
    deltas = payload["deltas"]
    n = min(deltas.shape[0], args.max_pairs)
    D = deltas[:n].float()
    D = D / (D.norm(dim=1, keepdim=True) + 1e-8)

    mu = D.mean(dim=0)
    mu = mu / (mu.norm() + 1e-8)
    cos_to_mu = D @ mu
    abs_cos_to_mu = cos_to_mu.abs()

    signs = torch.sign(cos_to_mu)
    signs[signs == 0] = 1
    D_aligned = D * signs.unsqueeze(1)

    m = args.pairwise_samples
    i = torch.randint(0, n, (m,))
    j = torch.randint(0, n, (m,))
    cos_ij = (D[i] * D[j]).sum(dim=1)

    q = min(10, n - 1) if n > 1 else 1
    if q == 1:
        var_explained = torch.tensor([1.0])
    else:
        _, S, _ = torch.pca_lowrank(D_aligned, q=q, center=True)
        var_explained = (S**2) / (S**2).sum()

    out = {
        "n": int(n),
        "cos_to_mean": summarize(cos_to_mu),
        "abs_cos_to_mean": summarize(abs_cos_to_mu),
        "pairwise_cos": summarize(cos_ij),
        "pca_var_explained_top10": [float(v.item()) for v in var_explained],
    }

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
