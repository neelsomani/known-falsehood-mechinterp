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
from run_behavioral_eval import load_consequences, load_splits, parse_ab_label


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
                # d: [d_model, k], assumed orthonormal columns
                coeffs = torch.matmul(v, d)
                proj = torch.matmul(coeffs, d.t())
                hs[:, tok, :] = v - spec.alpha * proj

        if rest is None:
            return hs
        return (hs,) + rest

    return hook


def prompt_last_token_index(input_ids: torch.Tensor) -> int:
    return input_ids.shape[1] - 1


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


def build_consequence_rows(
    data_path: Path,
    train_facts: set[str],
    train_templates: set[str],
    consequences: dict[tuple[str, str, str], dict[str, str]],
) -> list[dict[str, str]]:
    out = []
    with data_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            base_fact = row["base_fact"]
            template_id = row["template_id"]
            if base_fact not in train_facts:
                continue
            if template_id not in train_templates:
                continue

            key = (base_fact, row["proposition_id"], row["stance_label"])
            conseq = consequences.get(key)
            if not conseq:
                continue

            consequence_prompt = conseq["consequence_prompt"]
            if consequence_prompt:
                consequence_prompt = consequence_prompt[0].lower() + consequence_prompt[1:]
            full_question = (
                f"\"{row['statement']}\" According to the assumption above, {consequence_prompt}"
            )
            out.append(
                {
                    "id": row["id"],
                    "base_fact": base_fact,
                    "proposition_id": row["proposition_id"],
                    "stance_label": row["stance_label"],
                    "template_id": row["template_id"],
                    "statement": row["statement"],
                    "full_question": full_question,
                    "expected_answer": conseq["expected_answer"],
                }
            )
    return out


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


def run_consequence_eval(
    model,
    tokenizer,
    system: str | None,
    splits_path: Path,
    data_path: Path,
    consequences_path: Path,
    max_new_tokens: int,
    limit_total: int | None,
    show_prompts: bool,
    layer_idx: int,
    alpha: float,
    direction_for_row,
    requires_direction: bool,
    allowed_pairs: set[str],
    pair_index: dict[tuple[str, str, str], str],
) -> dict[str, float]:
    train_facts, train_templates = load_splits(splits_path)
    consequences = load_consequences(consequences_path)
    rows = build_consequence_rows(data_path, train_facts, train_templates, consequences)

    total = 0
    correct = 0
    unparsed = 0
    skipped_missing_delta = 0

    for row in rows:
        if limit_total is not None and total >= limit_total:
            break
        pair_key = (row["base_fact"], row["proposition_id"], row["template_id"])
        pair_id = pair_index.get(pair_key)
        if not pair_id or pair_id not in allowed_pairs:
            skipped_missing_delta += 1
            continue
        direction = direction_for_row(row, pair_id)
        if requires_direction and direction is None:
            skipped_missing_delta += 1
            continue

        full_prompt = build_prompt(
            tokenizer,
            system,
            row["full_question"],
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
            print("---- Consequence Task ----")
            print(row["full_question"])
            print(f"Model: {response}")
        parsed = parse_ab_label(response)
        if not parsed:
            unparsed += 1
        if parsed == row["expected_answer"]:
            correct += 1
        total += 1

    acc = correct / total if total else 0.0
    unparsed_pct = (unparsed / total * 100.0) if total else 0.0
    return {
        "acc": acc,
        "unparsed_pct": unparsed_pct,
        "skipped_missing_delta": skipped_missing_delta,
        "total": total,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline vs local vs averaged vs random delta ablations."
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
    parser.add_argument("--delta-pt", type=Path, required=True)
    parser.add_argument(
        "--pairs-jsonl",
        type=Path,
        default=Path("dataset/intervention_pairs.jsonl"),
        help="Path to paired prompts JSONL.",
    )
    parser.add_argument("--splits", default="dataset/splits.json", help="Path to splits.json.")
    parser.add_argument("--data", default="dataset/data.csv", help="Path to data.csv.")
    parser.add_argument(
        "--consequences", default="dataset/consequences.csv", help="Path to consequences.csv."
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--limit-consequence",
        type=int,
        default=None,
        help="Limit the number of consequence questions evaluated.",
    )
    parser.add_argument(
        "--show-prompts",
        action="store_true",
        help="Print prompts and model outputs as they are evaluated.",
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--layer", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--pca-ks",
        default="1,2,3",
        help="Comma-separated PCA subspace sizes to evaluate.",
    )
    parser.add_argument(
        "--pca-mode",
        choices=["centered", "uncentered"],
        default="centered",
        help="Use centered PCA or uncentered SVD on aligned deltas.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("dataset/avg_delta_eval.json"),
        help="Path to write JSON summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    payload = torch.load(args.delta_pt, map_location="cpu")
    deltas = payload["deltas"].to(dtype=torch_dtype)
    pair_ids = payload["pair_ids"]
    allowed_pairs = set(pair_ids)

    deltas_f = deltas.float()
    mean_delta_f = deltas_f.mean(dim=0)
    mean_delta_f = mean_delta_f / (mean_delta_f.norm() + 1e-8)
    rand = torch.randn_like(mean_delta_f)
    rand = rand / (rand.norm() + 1e-8)

    cos_to_mean = torch.matmul(deltas_f, mean_delta_f)
    signs = torch.sign(cos_to_mean)
    signs[signs == 0] = 1
    deltas_aligned = deltas_f * signs.unsqueeze(1)
    if args.pca_mode == "uncentered":
        if deltas_aligned.shape[0] == 1:
            pca_basis = mean_delta_f.unsqueeze(1)
        else:
            _, _, Vh = torch.linalg.svd(deltas_aligned, full_matrices=False)
            pca_basis = Vh[: min(10, Vh.shape[0])].t()
    else:
        q = min(10, deltas_aligned.shape[0] - 1) if deltas_aligned.shape[0] > 1 else 1
        if q == 1:
            pca_basis = mean_delta_f.unsqueeze(1)
        else:
            _, _, V = torch.pca_lowrank(deltas_aligned, q=q, center=True)
            pca_basis = V

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.eval()

    pairs = load_pairs(args.pairs_jsonl)
    pair_index = {}
    key_to_pairs: dict[tuple[str, str, str], set[str]] = {}
    for pair_id, pair in pairs.items():
        for row in pair.values():
            key = (row["base_fact"], row["proposition_id"], row["template_id"])
            key_to_pairs.setdefault(key, set()).add(pair_id)
            pair_index[key] = pair_id
    collisions = {k: v for k, v in key_to_pairs.items() if len(v) > 1}
    if collisions:
        sample = list(collisions.items())[:5]
        msg = ", ".join(f"{k} -> {sorted(v)}" for k, v in sample)
        raise ValueError(
            "pair_index collisions detected; key not unique. Example(s): " + msg
        )

    per_pair_lookup = {pid: deltas[i] for i, pid in enumerate(pair_ids)}

    def direction_local(_row, pair_id):
        return per_pair_lookup.get(pair_id)

    def direction_mean(_row, _pair_id):
        return mean_delta_f

    def direction_random(_row, _pair_id):
        return rand

    def direction_none(_row, _pair_id):
        return None

    results = {}
    evaluations = [
        ("baseline", direction_none, False),
        ("local_delta", direction_local, True),
        ("mean_delta", direction_mean, True),
        ("random_delta", direction_random, True),
    ]

    pca_ks = []
    for part in args.pca_ks.split(","):
        part = part.strip()
        if not part:
            continue
        pca_ks.append(int(part))
    for k in pca_ks:
        max_k = pca_basis.shape[1]
        if k <= 0 or k > max_k:
            raise ValueError(f"Invalid PCA k={k}; available={max_k}")
        basis = pca_basis[:, :k]

        def direction_pca(_row, _pair_id, basis=basis):
            return basis

        evaluations.append((f"pca_k{k}", direction_pca, True))

    for name, provider, requires_direction in evaluations:
        print(f"== {name} ==")
        metrics = run_consequence_eval(
            model,
            tokenizer,
            args.system,
            Path(args.splits),
            Path(args.data),
            Path(args.consequences),
            args.max_new_tokens,
            args.limit_consequence,
            args.show_prompts,
            args.layer,
            args.alpha,
            provider,
            requires_direction,
            allowed_pairs,
            pair_index,
        )
        results[name] = metrics
        print(
            f"  acc={metrics['acc']:.3f} "
            f"unparsed={metrics['unparsed_pct']:.2f}% "
            f"skipped={metrics['skipped_missing_delta']} "
            f"n={metrics['total']}"
        )

    output = {
        "layer": args.layer,
        "alpha": args.alpha,
        "delta_pt": str(args.delta_pt),
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
