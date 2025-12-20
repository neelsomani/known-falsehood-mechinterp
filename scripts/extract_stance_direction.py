#!/usr/bin/env python3
"""Extract and normalize stance direction from a cached probe."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a cached probe (.npz), undo StandardScaler, and L2-normalize."
        )
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("dataset/probe_cache"),
        help="Directory containing cached probe .npz files.",
    )
    parser.add_argument("--layer", type=int, default=15, help="Probe layer index.")
    parser.add_argument(
        "--position",
        choices=["entity", "final"],
        default="final",
        help="Activation position used for the probe.",
    )
    parser.add_argument(
        "--auto-select",
        action="store_true",
        help="Select layer/position from probe_aurocs.csv (max A minus C).",
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=Path("dataset/probe_aurocs.csv"),
        help="Path to probe_aurocs.csv used for auto selection.",
    )
    parser.add_argument(
        "--earliest-saturation",
        action="store_true",
        help="Pick earliest layer where Task A reaches its max (ties by A minus C).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Probe seed.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("dataset/stance_direction.npz"),
        help="Output .npz path.",
    )
    parser.add_argument(
        "--sanity-check",
        action="store_true",
        help="Run a quick sign check on Task A using activations and labels.",
    )
    parser.add_argument(
        "--acts",
        type=Path,
        default=Path("dataset/activations_test.pt"),
        help="Activations .pt path for sanity check.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("dataset/data.csv"),
        help="Data CSV with stance labels for sanity check.",
    )
    return parser.parse_args()


def select_best_site(metrics_csv: Path, earliest_saturation: bool) -> tuple[int, str]:
    with metrics_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"No rows in {metrics_csv}")

    # Build a map: (layer, position) -> {task: mean_auroc}
    site = {}
    for r in rows:
        key = (int(r["layer"]), r["position"])
        site.setdefault(key, {})[r["task"]] = float(r["mean_auroc"])

    # Filter to sites with Task A present.
    sites = []
    for (layer, position), task_map in site.items():
        if "A" not in task_map or "C" not in task_map:
            continue
        a = task_map["A"]
        c = task_map["C"]
        sites.append((layer, position, a, c, a - c))

    if not sites:
        raise ValueError(f"No Task A/C rows found in {metrics_csv}")

    if earliest_saturation:
        max_a = max(s[2] for s in sites)
        sat = [s for s in sites if s[2] == max_a]
        # Earliest layer, then prefer higher A-C as tie-break.
        sat.sort(key=lambda s: (s[0], -s[4]))
        best = sat[0]
    else:
        best = max(sites, key=lambda s: s[4])

    return best[0], best[1]


def load_activations(path: Path) -> tuple[np.ndarray, list[str]]:
    payload = torch.load(path, map_location="cpu")
    acts = payload["activations"]
    if isinstance(acts, torch.Tensor):
        acts = acts.detach().cpu().numpy()
    ids = payload["ids"]
    return acts, list(ids)


def load_data(path: Path) -> dict[str, dict[str, str]]:
    out = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[row["id"]] = row
    return out


def build_task_a_indices(ids: list[str], data_by_id: dict[str, dict[str, str]]):
    labels = []
    indices = []
    for i, row_id in enumerate(ids):
        row = data_by_id.get(row_id)
        if not row:
            continue
        stance = row["stance_label"]
        prop_id = row["proposition_id"]
        is_true = prop_id.endswith("__true")
        if not is_true or stance not in {"declared_true", "declared_false"}:
            continue
        label = 1 if stance == "declared_true" else 0
        indices.append(i)
        labels.append(label)
    return np.array(indices, dtype=np.int64), np.array(labels, dtype=np.int64)


def main() -> None:
    args = parse_args()
    if args.auto_select:
        layer, position = select_best_site(
            args.metrics_csv, args.earliest_saturation
        )
    else:
        layer, position = args.layer, args.position

    pos_idx = 0 if position == "entity" else 1

    cache_path = args.cache_dir / f"A_L{layer}_P{pos_idx}_S{args.seed}.npz"
    print("loading:", cache_path)
    z = np.load(cache_path)
    coef = z["coef"]  # shape (1, d_model)
    scale = z["scale"]  # shape (d_model,)

    # Undo StandardScaler: divide by scale in original feature space.
    w_raw = (coef[0] / scale).astype(np.float32)
    w = w_raw / np.linalg.norm(w_raw)

    np.savez(
        args.out,
        best_layer=layer,
        best_position=position,
        w=w,
        w_raw=w_raw,
        cache_path=str(cache_path),
    )
    print("saved:", args.out)
    print("w shape:", w.shape)
    print("position index:", pos_idx)

    if args.sanity_check:
        acts, ids = load_activations(args.acts)
        data_by_id = load_data(args.data)
        idx, labels = build_task_a_indices(ids, data_by_id)
        if idx.size == 0:
            raise ValueError("No Task A examples found for sanity check.")
        X = acts[idx, layer, pos_idx, :].astype(np.float32, copy=False)
        dots = X @ w
        mean_true = float(dots[labels == 1].mean())
        mean_false = float(dots[labels == 0].mean())
        print("sanity check (Task A):")
        print("  mean dot (declared_true):", mean_true)
        print("  mean dot (declared_false):", mean_false)
        if mean_true <= mean_false:
            print("  WARNING: sign looks flipped for Task A.")


if __name__ == "__main__":
    main()
