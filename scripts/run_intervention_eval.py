#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class InterventionSpec:
    layer: int
    token_index: int
    alpha: float
    w: torch.Tensor


def load_w(npz_path: Path, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    z = np.load(npz_path)
    w = torch.from_numpy(z["w"]).to(device=device, dtype=dtype)
    w = w / (w.norm() + 1e-8)
    return w


def make_projection_ablation_hook(spec: InterventionSpec):
    def hook(module, inputs, outputs):
        if isinstance(outputs, tuple):
            hs = outputs[0]
            rest = outputs[1:]
        else:
            hs = outputs
            rest = None

        tok = spec.token_index
        if tok < 0 or tok >= hs.shape[1]:
            return outputs

        v = hs[:, tok, :]
        w = spec.w if spec.w.device == hs.device else spec.w.to(hs.device)
        proj = torch.matmul(v, w)
        hs[:, tok, :] = v - spec.alpha * proj.unsqueeze(-1) * w.unsqueeze(0)

        if rest is None:
            return hs
        return (hs,) + rest

    return hook


def force_choice_last_token_logits(
    model,
    input_ids: torch.Tensor,
    choice_token_ids: list[int],
) -> torch.Tensor:
    with torch.no_grad():
        out = model(input_ids=input_ids)
        logits = out.logits[:, -1, :]
        return logits[:, choice_token_ids]


def pick_choice(logits_subset: torch.Tensor) -> int:
    return int(torch.argmax(logits_subset, dim=-1).item())


def parse_alphas(value: str) -> list[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


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


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description="Run stance-direction intervention evaluation.")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-70B-Instruct")
    ap.add_argument("--prompts-jsonl", type=Path, required=True)
    ap.add_argument("--w", default="dataset/stance_direction.npz")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--alphas", default="0,0.25,0.5,0.75,1.0,1.5,2.0")
    ap.add_argument("--direction", choices=["stance", "random", "random-orthogonal"], default="stance")
    ap.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("dataset/intervention_results.jsonl"))
    ap.add_argument("--flips-out", type=Path, default=Path("dataset/intervention_flips.jsonl"))
    ap.add_argument("--max-flips-per-alpha", type=int, default=10)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    w = load_w(Path(args.w), device=model.device, dtype=dtype)
    if args.direction == "random":
        w = torch.randn_like(w)
        w = w / (w.norm() + 1e-8)
    elif args.direction == "random-orthogonal":
        r = torch.randn_like(w)
        r = r / (r.norm() + 1e-8)
        r = r - torch.dot(r, w) * w
        w = r / (r.norm() + 1e-8)

    records = load_records(args.prompts_jsonl)
    if not records:
        raise ValueError("No prompts loaded.")

    alphas = parse_alphas(args.alphas)
    results = []
    baseline = {}

    for alpha in alphas:
        correct = 0
        total = 0
        for ex in records:
            prompt = ex["prompt"]
            tok_idx = int(ex["statement_final_token_index"])
            label = ex["label"]
            choices = ex["choices"]
            choice_token_ids = [
                tokenizer.encode(c, add_special_tokens=False) for c in choices
            ]
            if any(len(ids) != 1 for ids in choice_token_ids):
                raise ValueError(f"Choice tokens must be single-token: {choices!r}")
            choice_token_ids = [ids[0] for ids in choice_token_ids]

            input_ids = tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids.to(model.device)

            handle = None
            if alpha != 0.0:
                spec = InterventionSpec(
                    layer=args.layer,
                    token_index=tok_idx,
                    alpha=alpha,
                    w=w,
                )
                handle = model.model.layers[args.layer].register_forward_hook(
                    make_projection_ablation_hook(spec)
                )

            logits_subset = force_choice_last_token_logits(model, input_ids, choice_token_ids)
            pred_idx = pick_choice(logits_subset)
            pred_label = normalize_label(choices[pred_idx])
            gold_idx = pick_gold_index(choices, label)
            gold_label = normalize_label(choices[gold_idx])
            margin = compute_margin(logits_subset, gold_idx)
            is_correct = pred_idx == gold_idx

            if handle is not None:
                handle.remove()

            record = {
                "id": ex.get("id"),
                "task": ex.get("task"),
                "alpha": alpha,
                "direction": args.direction,
                "layer": args.layer,
                "token_index": tok_idx,
                "pred": pred_label,
                "gold": gold_label,
                "correct": bool(is_correct),
                "margin": float(margin),
            }
            results.append(record)
            if alpha == 0.0:
                baseline[ex.get("id")] = record
            correct += int(is_correct)
            total += 1

        acc = correct / total if total else 0.0
        print(f"alpha={alpha:.3f} acc={acc:.3f} ({correct}/{total})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    if 0.0 in alphas:
        flips = []
        for alpha in alphas:
            if alpha == 0.0:
                continue
            flipped = 0
            for r in results:
                if r["alpha"] != alpha:
                    continue
                base = baseline.get(r["id"])
                if not base:
                    continue
                if base["correct"] and not r["correct"]:
                    flips.append(
                        {
                            "id": r["id"],
                            "task": r["task"],
                            "alpha": alpha,
                            "pred_base": base["pred"],
                            "pred_alpha": r["pred"],
                            "gold": r["gold"],
                            "margin_base": base["margin"],
                            "margin_alpha": r["margin"],
                        }
                    )
                    flipped += 1
                if flipped >= args.max_flips_per_alpha:
                    break
        if flips:
            args.flips_out.parent.mkdir(parents=True, exist_ok=True)
            with args.flips_out.open("w", encoding="utf-8") as f:
                for r in flips:
                    f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
