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


def make_capture_hook(container: dict, key: str, token_index: int):
    def hook(module, inputs, outputs):
        hs = outputs[0] if isinstance(outputs, tuple) else outputs
        if token_index < 0 or token_index >= hs.shape[1]:
            raise ValueError("Capture token index out of range.")
        container[key] = hs[:, token_index, :].detach().cpu()
        return outputs

    return hook


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
            if d.ndim == 2:
                d = d[0]
            proj = torch.matmul(v, d)
            hs[:, tok, :] = v - spec.alpha * proj.unsqueeze(-1) * d.unsqueeze(0)

        if rest is None:
            return hs
        return (hs,) + rest

    return hook


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


def prompt_last_token_index(input_ids: torch.Tensor) -> int:
    return input_ids.shape[1] - 1


def compute_pair_deltas(
    model,
    tokenizer,
    layer_idx: int,
    pairs: dict[str, dict[str, dict]],
    pair_ids: list[str],
    log_every: int,
) -> dict[str, torch.Tensor]:
    deltas: dict[str, torch.Tensor] = {}
    total = len(pair_ids)
    for idx, pair_id in enumerate(pair_ids, start=1):
        pair = pairs[pair_id]
        if "declared_true" not in pair or "declared_false" not in pair:
            continue
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
        true_last = prompt_last_token_index(true_input)
        false_last = prompt_last_token_index(false_input)
        h_true = run_and_capture(model, model.model.layers[layer_idx], true_input, true_last)
        h_false = run_and_capture(model, model.model.layers[layer_idx], false_input, false_last)
        delta = (h_true - h_false).squeeze(0)
        delta = delta / (delta.norm() + 1e-8)
        deltas[pair_id] = delta.cpu()
        if log_every and (idx % log_every == 0 or idx == total):
            print(f"  delta pairs: {idx}/{total}")
    if not deltas:
        raise ValueError("No valid pairs available to build per-pair deltas.")
    return deltas


def save_pair_deltas(path: Path, deltas: dict[str, torch.Tensor], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pair_ids = sorted(deltas.keys())
    stacked = torch.stack([deltas[pair_id] for pair_id in pair_ids], dim=0)
    tmp_path = path.with_suffix(".tmp.pt")
    torch.save({"pair_ids": pair_ids, "deltas": stacked, "meta": meta}, tmp_path)
    tmp_path.replace(path)


def load_pair_deltas(path: Path, dtype: torch.dtype) -> tuple[dict[str, torch.Tensor], dict]:
    payload = torch.load(path, map_location="cpu")
    pair_ids = payload["pair_ids"]
    deltas = payload["deltas"].to(dtype=dtype)
    meta = payload.get("meta", {})
    return {pair_id: deltas[i] for i, pair_id in enumerate(pair_ids)}, meta


def generate_with_ablation(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    layer_idx: int,
    tok_idx: int,
    alpha: float,
    w: torch.Tensor,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    handle = model.model.layers[layer_idx].register_forward_hook(
        make_delta_ablate_hook(
            AblateSpec(token_index=tok_idx, alpha=alpha, direction=w)
        )
    )
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    handle.remove()
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


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


def run_consequence_task(
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
    deltas_by_pair: dict[str, torch.Tensor],
    pair_index: dict[tuple[str, str, str], str],
    allowed_pairs: set[str],
) -> dict[str, float]:
    train_facts, train_templates = load_splits(splits_path)
    consequences = load_consequences(consequences_path)
    rows = build_consequence_rows(data_path, train_facts, train_templates, consequences)

    total = 0
    correct = 0
    unparsed = 0
    breakdown: dict[tuple[str, str, str], dict[str, int]] = {}

    skipped_missing_delta = 0
    for row in rows:
        if limit_total is not None and total >= limit_total:
            break
        pair_key = (row["base_fact"], row["proposition_id"], row["template_id"])
        pair_id = pair_index.get(pair_key)
        if not pair_id or pair_id not in allowed_pairs:
            skipped_missing_delta += 1
            continue
        delta = deltas_by_pair.get(pair_id)
        if delta is None:
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
        response = generate_with_ablation(
            model,
            tokenizer,
            full_prompt,
            max_new_tokens=max_new_tokens,
            layer_idx=layer_idx,
            tok_idx=tok_idx,
            alpha=alpha,
            w=delta,
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
        template_id = row["template_id"]
        if template_id == "T_BARE":
            family = "BARE"
        else:
            family = template_id.rsplit("_", 1)[-1]
        prop_kind = "X_true" if row["proposition_id"].endswith("__true") else "X_false"
        stance_label = row["stance_label"]
        key = (family, stance_label, prop_kind)
        if key not in breakdown:
            breakdown[key] = {"total": 0, "correct": 0, "unparsed": 0}
        breakdown[key]["total"] += 1
        if not parsed:
            breakdown[key]["unparsed"] += 1
        if parsed == row["expected_answer"]:
            breakdown[key]["correct"] += 1
        total += 1

    acc = correct / total if total else 0.0
    unparsed_pct = (unparsed / total * 100.0) if total else 0.0

    print("Consequence task (train facts/templates; ablated):")
    print(f"  Accuracy: {acc:.3f} ({correct}/{total})")
    print(f"  Unparsed percent: {unparsed_pct:.2f}% ({unparsed}/{total})")
    print(f"  Skipped (missing delta): {skipped_missing_delta}")
    print("Consequence breakdown by template family:")
    for key in sorted(breakdown.keys()):
        family, stance_label, prop_kind = key
        stats = breakdown[key]
        bucket_total = stats["total"]
        if not bucket_total:
            bucket_acc = 0.0
            bucket_unparsed_pct = 0.0
        else:
            bucket_acc = stats["correct"] / bucket_total
            bucket_unparsed_pct = stats["unparsed"] / bucket_total * 100.0
        print(
            "  "
            f"{stance_label}({prop_kind})"
            f" | family {family}"
            f": acc {bucket_acc:.3f} ({stats['correct']}/{bucket_total})"
            f", unparsed {bucket_unparsed_pct:.2f}% ({stats['unparsed']}/{bucket_total})"
        )
    return {
        "acc": acc,
        "unparsed_pct": unparsed_pct,
        "skipped_missing_delta": skipped_missing_delta,
    }


def parse_layers(spec: str, num_layers: int) -> list[int]:
    spec = spec.strip().lower()
    if spec in {"all", ""}:
        return list(range(num_layers))
    if "," in spec:
        out = []
        for part in spec.split(","):
            part = part.strip()
            if part == "last":
                out.append(num_layers - 1)
            elif "-" in part:
                start_s, end_s = part.split("-", 1)
                start = int(start_s)
                end = int(end_s)
                out.extend(range(start, end + 1))
            else:
                out.append(int(part))
        return sorted(set(out))
    if "-" in spec:
        start_s, end_s = spec.split("-", 1)
        return list(range(int(start_s), int(end_s) + 1))
    if spec == "last":
        return [num_layers - 1]
    return [int(spec)]


def load_completed_layers(path: Path) -> set[int]:
    completed = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") == "done":
                completed.add(int(row["layer"]))
    return completed


def append_result(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-layer delta directions and run behavioral eval ablations."
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument(
        "--layers",
        default="all",
        help="Layer list: 'all', 'last', '0-10', or '0,2,4'.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("dataset/delta_directions"),
        help="Directory to store per-layer delta files.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("dataset/layerwise_delta_eval.jsonl"),
        help="JSONL results output (one row per layer).",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="Log progress every N pairs during delta computation.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute deltas and rerun eval even if outputs exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.eval()

    pairs = load_pairs(args.pairs_jsonl)
    all_pair_ids = list(pairs.keys())
    random.seed(args.seed)
    random.shuffle(all_pair_ids)
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

    train_facts, train_templates = load_splits(Path(args.splits))
    consequences = load_consequences(Path(args.consequences))
    base_rows = build_consequence_rows(Path(args.data), train_facts, train_templates, consequences)
    required_pairs = set()
    for row in base_rows:
        key = (row["base_fact"], row["proposition_id"], row["template_id"])
        pair_id = pair_index.get(key)
        if pair_id:
            required_pairs.add(pair_id)

    pair_ids = [pid for pid in all_pair_ids if pid in required_pairs]
    if args.max_pairs is not None:
        pair_ids = pair_ids[: args.max_pairs]
    allowed_pairs = set(pair_ids)

    layers = parse_layers(args.layers, len(model.model.layers))
    completed = load_completed_layers(args.results)

    print(f"Total layers: {len(model.model.layers)}")
    print(f"Running layers: {layers}")
    print(f"Using {len(pair_ids)} pairs for delta estimation/eval")
    if completed and not args.force:
        print(f"Resuming; already completed layers: {sorted(completed)}")

    for idx, layer_idx in enumerate(layers, start=1):
        if layer_idx in completed and not args.force:
            print(f"[{idx}/{len(layers)}] Layer {layer_idx}: already done, skipping.")
            continue

        delta_path = args.out_dir / f"delta_layer_{layer_idx:03d}.pt"
        if args.force or not delta_path.exists():
            print(f"[{idx}/{len(layers)}] Layer {layer_idx}: computing per-pair deltas")
            deltas_by_pair = compute_pair_deltas(
                model,
                tokenizer,
                layer_idx,
                pairs,
                pair_ids,
                log_every=args.log_every,
            )
            meta = {
                "layer": layer_idx,
                "max_pairs": args.max_pairs,
                "seed": args.seed,
                "token_position": "last",
                "pair_count": len(deltas_by_pair),
            }
            save_pair_deltas(delta_path, deltas_by_pair, meta)
            print(f"  saved {delta_path}")
        else:
            print(f"[{idx}/{len(layers)}] Layer {layer_idx}: using existing {delta_path}")
        deltas_by_pair, meta = load_pair_deltas(delta_path, dtype=torch_dtype)
        if meta:
            expected = {
                "layer": layer_idx,
                "max_pairs": args.max_pairs,
                "seed": args.seed,
                "token_position": "last",
                "pair_count": len(deltas_by_pair),
            }
            for key, value in expected.items():
                if meta.get(key) != value:
                    print(f"  warning: delta meta mismatch for {key}: {meta.get(key)} != {value}")

        print(f"[{idx}/{len(layers)}] Layer {layer_idx}: skipping truth eval (no paired deltas)")
        print(f"[{idx}/{len(layers)}] Layer {layer_idx}: running consequence eval")
        consequence_metrics = run_consequence_task(
            model,
            tokenizer,
            args.system,
            Path(args.splits),
            Path(args.data),
            Path(args.consequences),
            args.max_new_tokens,
            args.limit_consequence,
            args.show_prompts,
            layer_idx,
            args.alpha,
            deltas_by_pair,
            pair_index,
            allowed_pairs,
        )
        result = {
            "status": "done",
            "layer": layer_idx,
            "delta_path": str(delta_path),
            "alpha": args.alpha,
            "max_pairs": args.max_pairs,
            "seed": args.seed,
            "token_position": "last",
            "consequence_metrics": consequence_metrics,
        }
        append_result(args.results, result)
        print(f"[{idx}/{len(layers)}] Layer {layer_idx}: results saved to {args.results}")


if __name__ == "__main__":
    main()
