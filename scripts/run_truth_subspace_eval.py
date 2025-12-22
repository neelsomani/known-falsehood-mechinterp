#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt_utils import build_prompt
from run_behavioral_eval import parse_truth_label


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
            d = spec.direction.to(device=v.device, dtype=v.dtype)
            if d.ndim == 1:
                proj = torch.matmul(v, d)
                hs[:, tok, :] = v - spec.alpha * proj.unsqueeze(-1) * d.unsqueeze(0)
            else:
                coeffs = torch.matmul(v, d)
                proj = torch.matmul(coeffs, d.t())
                hs[:, tok, :] = v - spec.alpha * proj

        if rest is None:
            return hs
        return (hs,) + rest

    return hook


def prompt_last_token_index(input_ids: torch.Tensor) -> int:
    return input_ids.shape[1] - 1


def generate(
    model,
    tokenizer,
    inputs,
    max_new_tokens: int,
    layer_idx: int,
    tok_idx: int,
    alpha: float,
    direction: torch.Tensor | None,
) -> str:
    handle = None
    if direction is not None:
        handle = model.model.layers[layer_idx].register_forward_hook(
            make_delta_ablate_hook(
                AblateSpec(token_index=tok_idx, alpha=alpha, direction=direction)
            )
        )
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    if handle is not None:
        handle.remove()
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def run_truth_eval(
    model,
    tokenizer,
    system: str | None,
    facts_path: Path,
    max_new_tokens: int,
    limit_per_class: int | None,
    show_prompts: bool,
    layer_idx: int,
    alpha: float,
    direction: torch.Tensor | None,
) -> dict[str, float]:
    true_total = 0
    false_total = 0
    true_correct = 0
    false_correct = 0
    unknown_count = 0

    with facts_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for label, fact in (("True", row["True Fact"]), ("False", row["False Fact"])):
                if limit_per_class is not None:
                    if label == "True" and true_total >= limit_per_class:
                        continue
                    if label == "False" and false_total >= limit_per_class:
                        continue
                prompt = f"\"{fact}\"\nIs the statement true? Answer True/False/Unknown."
                full_prompt = build_prompt(
                    tokenizer,
                    system,
                    prompt,
                    add_generation_prompt=True,
                )
                inputs = tokenizer(full_prompt, return_tensors="pt")
                tok_idx = prompt_last_token_index(inputs["input_ids"])
                response = generate(
                    model,
                    tokenizer,
                    inputs,
                    max_new_tokens=max_new_tokens,
                    layer_idx=layer_idx,
                    tok_idx=tok_idx,
                    alpha=alpha,
                    direction=direction,
                )
                if show_prompts:
                    print("---- Truth Task ----")
                    print(prompt)
                    print(f"Model: {response}")
                parsed = parse_truth_label(response)
                if parsed == "Unknown":
                    unknown_count += 1
                if label == "True":
                    true_total += 1
                    if parsed == "True":
                        true_correct += 1
                else:
                    false_total += 1
                    if parsed == "False":
                        false_correct += 1
            if limit_per_class is not None and true_total >= limit_per_class and false_total >= limit_per_class:
                break

    total = true_total + false_total
    true_acc = true_correct / true_total if true_total else 0.0
    false_acc = false_correct / false_total if false_total else 0.0
    overall_acc = (true_correct + false_correct) / total if total else 0.0
    unknown_pct = (unknown_count / total * 100.0) if total else 0.0
    return {
        "true_acc": true_acc,
        "false_acc": false_acc,
        "overall_acc": overall_acc,
        "unknown_pct": unknown_pct,
        "total": total,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run truth eval with PCA subspace ablation."
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-70B-Instruct",
        help="Hugging Face model id.",
    )
    parser.add_argument(
        "--system",
        default=(
            "You are a helpful agent. You are a helpful agent. By default, evaluate statements "
            "using your general world knowledge. When a statement is presented as an explicit "
            "assumption for the sake of reasoning, condition your reasoning on that assumption "
            "rather than on world knowledge. Do not treat statements as assumptions unless the "
            "user's phrasing indicates they are to be taken as such. Answer the user's prompt "
            "in 1 character (A/B)."
        ),
        help="Optional system prompt.",
    )
    parser.add_argument("--facts", default="dataset/facts.csv", help="Path to facts.csv.")
    parser.add_argument("--delta-pt", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--limit-truth-per-class",
        type=int,
        default=None,
        help="Limit the number of true and false facts evaluated.",
    )
    parser.add_argument(
        "--show-prompts",
        action="store_true",
        help="Print prompts and model outputs as they are evaluated.",
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--layer", type=int, default=40)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument(
        "--pca-mode",
        choices=["uncentered"],
        default="uncentered",
        help="Use uncentered SVD on aligned deltas.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("dataset/truth_subspace_eval.json"),
        help="Path to write JSON summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    payload = torch.load(args.delta_pt, map_location="cpu")
    deltas = payload["deltas"].float()
    mean_delta = deltas.mean(dim=0)
    mean_delta = mean_delta / (mean_delta.norm() + 1e-8)
    cos_to_mean = torch.matmul(deltas, mean_delta)
    signs = torch.sign(cos_to_mean)
    signs[signs == 0] = 1
    deltas_aligned = deltas * signs.unsqueeze(1)

    if args.pca_mode == "uncentered":
        if deltas_aligned.shape[0] == 1:
            basis = mean_delta.unsqueeze(1)
        else:
            _, _, Vh = torch.linalg.svd(deltas_aligned, full_matrices=False)
            basis = Vh[: args.k].t()
    else:
        raise ValueError(f"Unsupported pca_mode: {args.pca_mode}")

    if basis.shape[1] < args.k:
        raise ValueError(f"Not enough PCA components for k={args.k}.")

    rand = torch.randn_like(basis)
    q, _ = torch.linalg.qr(rand, mode="reduced")
    rand_basis = q[:, : args.k]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.eval()

    results = {}
    for name, direction in [
        ("baseline", None),
        ("pca_subspace", basis),
        ("random_subspace", rand_basis),
    ]:
        print(f"== {name} ==")
        metrics = run_truth_eval(
            model,
            tokenizer,
            args.system,
            Path(args.facts),
            args.max_new_tokens,
            args.limit_truth_per_class,
            args.show_prompts,
            args.layer,
            args.alpha,
            direction,
        )
        results[name] = metrics
        print(
            f"  overall_acc={metrics['overall_acc']:.3f} "
            f"unknown={metrics['unknown_pct']:.2f}% "
            f"n={metrics['total']}"
        )

    output = {
        "layer": args.layer,
        "k": args.k,
        "alpha": args.alpha,
        "delta_pt": str(args.delta_pt),
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
