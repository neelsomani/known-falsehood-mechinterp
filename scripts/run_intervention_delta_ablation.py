#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def make_capture_hook(container: dict, key: str, token_index: int):
    def hook(module, inputs, outputs):
        hs = outputs[0] if isinstance(outputs, tuple) else outputs
        if token_index < 0 or token_index >= hs.shape[1]:
            raise ValueError("Capture token index out of range.")
        container[key] = hs[:, token_index, :].detach().cpu()
        return outputs

    return hook


@dataclass(frozen=True)
class AblateSpec:
    token_index: int
    alpha: float
    direction: torch.Tensor


def make_delta_ablate_hook(spec: AblateSpec):
    def hook(module, inputs, outputs):
        if isinstance(outputs, tuple):
            hs = outputs[0]
            rest = outputs[1:]
        else:
            hs = outputs
            rest = None

        tok = spec.token_index
        if 0 <= tok < hs.shape[1]:
            v = hs[:, tok, :]
            d = spec.direction.to(v.device)
            if d.ndim == 2:
                d = d[0]
            d = d / (d.norm() + 1e-8)
            proj = torch.matmul(v, d)
            hs[:, tok, :] = v - spec.alpha * proj.unsqueeze(-1) * d.unsqueeze(0)

        if rest is None:
            return hs
        return (hs,) + rest

    return hook


def normalize_label(text: str) -> str:
    stripped = text.strip()
    if stripped.lower() in {"a", "b"}:
        return stripped.upper()
    return stripped.capitalize()


def pick_gold_index(choices: list[str], label: str) -> int:
    norm_label = normalize_label(label)
    for idx, choice in enumerate(choices):
        if normalize_label(choice) == norm_label:
            return idx
    raise ValueError(f"Label {label!r} not found in choices {choices!r}.")


def compute_margin(logits_subset: torch.Tensor, gold_idx: int) -> float:
    logits = logits_subset.squeeze(0)
    gold_logit = logits[gold_idx].item()
    other = torch.cat([logits[:gold_idx], logits[gold_idx + 1 :]])
    if other.numel() == 0:
        return 0.0
    return gold_logit - torch.max(other).item()


def pick_pred_label(logits_subset: torch.Tensor, choices: list[str]) -> str:
    pred_idx = int(torch.argmax(logits_subset, dim=-1).item())
    return normalize_label(choices[pred_idx])


def load_pairs(path: Path) -> dict[str, dict[str, dict]]:
    pairs: dict[str, dict[str, dict]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            pair_id = row["pair_id"]
            stance = row["stance_label"]
            pairs.setdefault(pair_id, {})[stance] = row
    return pairs


def run_and_capture(model, layer_module, input_ids: torch.Tensor, tok_idx: int) -> torch.Tensor:
    cache = {}
    handle = layer_module.register_forward_hook(make_capture_hook(cache, "x", tok_idx))
    with torch.no_grad():
        _ = model(input_ids=input_ids)
    handle.remove()
    return cache["x"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Delta-direction ablation for stance pairs.")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-70B-Instruct")
    ap.add_argument(
        "--pairs-jsonl",
        type=Path,
        default=Path("dataset/intervention_pairs.jsonl"),
        help="Path to paired prompts JSONL.",
    )
    ap.add_argument("--layer", default="last", help="Layer index or 'last'.")
    ap.add_argument(
        "--direction",
        choices=["true-to-false", "false-to-true", "both"],
        default="both",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Strength of delta projection removal.",
    )
    ap.add_argument(
        "--delta-mode",
        choices=["per-pair", "global"],
        default="per-pair",
        help="Use per-pair delta or a global average delta.",
    )
    ap.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-pairs", type=int, default=50)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("dataset/intervention_delta_ablation.jsonl"),
    )
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    pairs = load_pairs(args.pairs_jsonl)
    pair_ids = list(pairs.keys())
    random.shuffle(pair_ids)
    pair_ids = pair_ids[: args.max_pairs]

    layer_idx = args.layer
    if str(layer_idx).lower() == "last":
        layer_idx = len(model.model.layers) - 1
    layer_idx = int(layer_idx)

    global_delta = None
    if args.delta_mode == "global":
        deltas = []
        for pair_id in pair_ids:
            pair = pairs[pair_id]
            if "declared_true" not in pair or "declared_false" not in pair:
                continue
            true_row = pair["declared_true"]
            false_row = pair["declared_false"]
            if true_row["choices"] != false_row["choices"]:
                continue
            true_input = tokenizer(
                true_row["prompt"],
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids
            false_input = tokenizer(
                false_row["prompt"],
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids
            true_last = true_input.shape[1] - 1
            false_last = false_input.shape[1] - 1
            h_true = run_and_capture(model, model.model.layers[layer_idx], true_input, true_last)
            h_false = run_and_capture(model, model.model.layers[layer_idx], false_input, false_last)
            delta = (h_true - h_false).squeeze(0)
            deltas.append(delta)
        if not deltas:
            raise ValueError("No valid pairs available to build global delta.")
        global_delta = torch.stack(deltas, dim=0).mean(dim=0)

    outputs = []
    for pair_id in pair_ids:
        pair = pairs[pair_id]
        if "declared_true" not in pair or "declared_false" not in pair:
            continue
        directions = []
        if args.direction in {"true-to-false", "both"}:
            directions.append(("declared_true", "declared_false"))
        if args.direction in {"false-to-true", "both"}:
            directions.append(("declared_false", "declared_true"))

        for source_stance, target_stance in directions:
            source = pair[source_stance]
            target = pair[target_stance]

            if source["choices"] != target["choices"]:
                raise ValueError("Choice set mismatch within pair.")

            target_input = tokenizer(
                target["prompt"],
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids

            target_last = target_input.shape[1] - 1

            choices = target["choices"]
            choice_token_ids = [
                tokenizer.encode(c, add_special_tokens=False) for c in choices
            ]
            if any(len(ids) != 1 for ids in choice_token_ids):
                raise ValueError(f"Choice tokens must be single-token: {choices!r}")
            choice_token_ids = [ids[0] for ids in choice_token_ids]
            gold_idx = pick_gold_index(choices, target["label"])

            if args.delta_mode == "per-pair":
                true_row = pair["declared_true"]
                false_row = pair["declared_false"]
                true_input = tokenizer(
                    true_row["prompt"],
                    return_tensors="pt",
                    add_special_tokens=False,
                ).input_ids
                false_input = tokenizer(
                    false_row["prompt"],
                    return_tensors="pt",
                    add_special_tokens=False,
                ).input_ids
                true_last = true_input.shape[1] - 1
                false_last = false_input.shape[1] - 1
                h_true = run_and_capture(
                    model,
                    model.model.layers[layer_idx],
                    true_input,
                    true_last,
                )
                h_false = run_and_capture(
                    model,
                    model.model.layers[layer_idx],
                    false_input,
                    false_last,
                )
                delta = (h_true - h_false).squeeze(0)
            else:
                delta = global_delta

            with torch.no_grad():
                base_out = model(input_ids=target_input)
            base_logits = base_out.logits[:, -1, :]
            base_subset = base_logits[:, choice_token_ids]
            base_margin = compute_margin(base_subset, gold_idx)
            base_pred = pick_pred_label(base_subset, choices)
            gold_label = normalize_label(target["label"])
            base_correct = base_pred == gold_label

            patch_handle = model.model.layers[layer_idx].register_forward_hook(
                make_delta_ablate_hook(
                    AblateSpec(
                        token_index=target_last,
                        alpha=args.alpha,
                        direction=delta,
                    )
                )
            )
            with torch.no_grad():
                ablated_out = model(input_ids=target_input)
            patch_handle.remove()

            ablated_logits = ablated_out.logits[:, -1, :]
            ablated_subset = ablated_logits[:, choice_token_ids]
            ablated_margin = compute_margin(ablated_subset, gold_idx)
            ablated_pred = pick_pred_label(ablated_subset, choices)
            ablated_correct = ablated_pred == gold_label

            outputs.append(
                {
                    "pair_id": pair_id,
                    "source_stance": source_stance,
                    "target_stance": target_stance,
                    "layer": layer_idx,
                    "delta_mode": args.delta_mode,
                    "alpha": args.alpha,
                    "base_margin": base_margin,
                    "ablated_margin": ablated_margin,
                    "delta_margin": ablated_margin - base_margin,
                    "base_pred": base_pred,
                    "ablated_pred": ablated_pred,
                    "gold": gold_label,
                    "base_correct": base_correct,
                    "ablated_correct": ablated_correct,
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in outputs:
            f.write(json.dumps(row) + "\n")

    deltas = [row["delta_margin"] for row in outputs]
    mean_delta = float(sum(deltas) / max(1, len(deltas)))
    mean_abs = float(sum(abs(d) for d in deltas) / max(1, len(deltas)))
    print(f"n={len(outputs)}")
    print(f"delta_margin mean={mean_delta:.6f} mean_abs={mean_abs:.6f}")


if __name__ == "__main__":
    main()
