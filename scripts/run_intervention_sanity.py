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
        if 0 <= tok < hs.shape[1]:
            v = hs[:, tok, :]
            w = spec.w if spec.w.device == hs.device else spec.w.to(hs.device)
            proj = torch.matmul(v, w)
            hs[:, tok, :] = v - spec.alpha * proj.unsqueeze(-1) * w.unsqueeze(0)

        if rest is None:
            return hs
        return (hs,) + rest

    return hook


def make_capture_hook(container: dict, key: str, token_index: int):
    def hook(module, inputs, outputs):
        if isinstance(outputs, tuple):
            hs = outputs[0]
        else:
            hs = outputs
        tok = token_index
        if tok < 0 or tok >= hs.shape[1]:
            raise ValueError("Capture token index out of range.")
        container[key] = hs[:, tok, :].detach()
        return outputs

    return hook


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
    ap = argparse.ArgumentParser(
        description="Sanity-check whether ablation affects last-token state and logits."
    )
    ap.add_argument("--model", default="meta-llama/Llama-3.1-70B-Instruct")
    ap.add_argument("--prompts-jsonl", type=Path, required=True)
    ap.add_argument("--w", default="dataset/stance_direction.npz")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-examples", type=int, default=50)
    ap.add_argument(
        "--use-last-token",
        action="store_true",
        help="Override statement_final_token_index and ablate at the final prompt token.",
    )
    ap.add_argument("--out", type=Path, default=Path("dataset/intervention_sanity.jsonl"))
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

    records = [r for r in load_records(args.prompts_jsonl) if r.get("task") == "consequence"]
    if not records:
        raise ValueError("No consequence prompts found.")

    random.shuffle(records)
    records = records[: args.max_examples]

    outputs = []
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
        last_tok = input_ids.shape[1] - 1
        if args.use_last_token:
            tok_idx = last_tok

        gold_idx = pick_gold_index(choices, label)

        base_cache = {}
        base_hook = model.model.layers[args.layer].register_forward_hook(
            make_capture_hook(base_cache, "last", last_tok)
        )
        with torch.no_grad():
            base_out = model(input_ids=input_ids)
        base_hook.remove()
        base_logits = base_out.logits[:, -1, :]
        base_margin = compute_margin(base_logits[:, choice_token_ids], gold_idx)
        base_proj = torch.matmul(base_cache["last"], w).item()

        ablate_cache = {}
        ablate_hook = model.model.layers[args.layer].register_forward_hook(
            make_projection_ablation_hook(
                InterventionSpec(
                    layer=args.layer,
                    token_index=tok_idx,
                    alpha=args.alpha,
                    w=w,
                )
            )
        )
        ablate_capture = model.model.layers[args.layer].register_forward_hook(
            make_capture_hook(ablate_cache, "last", last_tok)
        )
        with torch.no_grad():
            ablate_out = model(input_ids=input_ids)
        ablate_capture.remove()
        ablate_hook.remove()

        ablate_logits = ablate_out.logits[:, -1, :]
        ablate_margin = compute_margin(ablate_logits[:, choice_token_ids], gold_idx)
        ablate_proj = torch.matmul(ablate_cache["last"], w).item()

        outputs.append(
            {
                "id": ex.get("id"),
                "alpha": args.alpha,
                "layer": args.layer,
                "statement_token_index": tok_idx,
                "last_token_index": last_tok,
                "margin_base": base_margin,
                "margin_ablate": ablate_margin,
                "delta_margin": ablate_margin - base_margin,
                "proj_base": base_proj,
                "proj_ablate": ablate_proj,
                "delta_proj": ablate_proj - base_proj,
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in outputs:
            f.write(json.dumps(row) + "\n")

    deltas = [row["delta_margin"] for row in outputs]
    proj_deltas = [row["delta_proj"] for row in outputs]
    mean_margin = float(sum(deltas) / max(1, len(deltas)))
    mean_abs_margin = float(sum(abs(d) for d in deltas) / max(1, len(deltas)))
    mean_proj = float(sum(proj_deltas) / max(1, len(proj_deltas)))
    mean_abs_proj = float(sum(abs(d) for d in proj_deltas) / max(1, len(proj_deltas)))
    print(f"n={len(outputs)}")
    print(f"delta_margin mean={mean_margin:.6f} mean_abs={mean_abs_margin:.6f}")
    print(f"delta_proj mean={mean_proj:.6f} mean_abs={mean_abs_proj:.6f}")


if __name__ == "__main__":
    main()
