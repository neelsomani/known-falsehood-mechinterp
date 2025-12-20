#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from transformers import AutoTokenizer

from prompt_utils import build_prompt, find_statement_span, spans_to_token_index
from run_behavioral_eval import (
    build_consequence_rows,
    load_consequences,
    load_splits,
)


TRUTH_SYSTEM_PROMPT = (
    "You are a helpful agent. By default, evaluate statements "
    "using your general world knowledge. When a statement is presented as an explicit "
    "assumption for the sake of reasoning, condition your reasoning on that assumption "
    "rather than on world knowledge. Do not treat statements as assumptions unless the "
    "user's phrasing indicates they are to be taken as such. Answer the user's prompt "
    "in 1 character (A/B)."
)


def statement_final_token_index(tokenizer, prompt: str, statement: str) -> int:
    tokenized = tokenizer(
        prompt,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    offsets = tokenized["offset_mapping"]
    if offsets and isinstance(offsets[0], (list, tuple)) and len(offsets[0]) == 2:
        normalized = offsets
    else:
        normalized = list(offsets[0])
    offsets = normalized
    statement_span = find_statement_span(prompt, statement)
    final_tok = spans_to_token_index(offsets, statement_span)
    if final_tok is None:
        raise ValueError("Could not locate final token index for statement.")
    return final_tok


def build_truth_prompts(
    tokenizer,
    system: str,
    facts_path: Path,
    limit_per_class: int | None,
    seed: int,
) -> list[dict]:
    rows = []
    random.seed(seed)
    true_rows = []
    false_rows = []
    with facts_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            true_rows.append(("True", row["True Fact"]))
            false_rows.append(("False", row["False Fact"]))
    if limit_per_class is not None:
        true_rows = true_rows[:limit_per_class]
        false_rows = false_rows[:limit_per_class]
    for label, fact in true_rows + false_rows:
        prompt_text = f"\"{fact}\"\nIs the statement true? Answer True/False/Unknown."
        full_prompt = build_prompt(
            tokenizer,
            system,
            prompt_text,
            add_generation_prompt=True,
        )
        tok_idx = statement_final_token_index(tokenizer, full_prompt, fact)
        rows.append(
            {
                "id": f"truth::{fact}",
                "task": "truth",
                "prompt": full_prompt,
                "statement": fact,
                "statement_final_token_index": tok_idx,
                "label": label,
                "choices": [" True", " False", " Unknown"],
            }
        )
    return rows


def build_consequence_prompts(
    tokenizer,
    system: str,
    splits_path: Path,
    data_path: Path,
    consequences_path: Path,
    limit_total: int | None,
    seed: int,
) -> list[dict]:
    train_facts, train_templates = load_splits(splits_path)
    consequences = load_consequences(consequences_path)
    rows = build_consequence_rows(data_path, train_facts, train_templates, consequences)
    if limit_total is not None:
        rows = rows[:limit_total]
    random.seed(seed)
    prompts = []
    for row in rows:
        prompt_text = row["full_question"]
        full_prompt = build_prompt(
            tokenizer,
            system,
            prompt_text,
            add_generation_prompt=True,
        )
        statement = row["statement"]
        tok_idx = statement_final_token_index(tokenizer, full_prompt, statement)
        prompts.append(
            {
                "id": row["id"],
                "task": "consequence",
                "prompt": full_prompt,
                "statement": statement,
                "statement_final_token_index": tok_idx,
                "label": row["expected_answer"],
                "choices": [" A", " B"],
            }
        )
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build prompts JSONL with statement-final token indices for interventions."
    )
    parser.add_argument("--model", default="meta-llama/Llama-3.1-70B-Instruct")
    parser.add_argument(
        "--system",
        default=TRUTH_SYSTEM_PROMPT,
        help="System prompt for building chat prompts.",
    )
    parser.add_argument("--task", choices=["truth", "consequence", "all"], default="all")
    parser.add_argument("--facts", default="dataset/facts.csv")
    parser.add_argument("--splits", default="dataset/splits.json")
    parser.add_argument("--data", default="dataset/data.csv")
    parser.add_argument("--consequences", default="dataset/consequences.csv")
    parser.add_argument("--limit-truth-per-class", type=int, default=None)
    parser.add_argument("--limit-consequence", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("dataset/intervention_prompts.jsonl"),
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)

    prompts = []
    if args.task in {"truth", "all"}:
        prompts.extend(
            build_truth_prompts(
                tokenizer,
                args.system,
                Path(args.facts),
                args.limit_truth_per_class,
                args.seed,
            )
        )
    if args.task in {"consequence", "all"}:
        prompts.extend(
            build_consequence_prompts(
                tokenizer,
                args.system,
                Path(args.splits),
                Path(args.data),
                Path(args.consequences),
                args.limit_consequence,
                args.seed,
            )
        )

    if not prompts:
        raise ValueError("No prompts generated.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in prompts:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(prompts)} prompts to {args.out}")


if __name__ == "__main__":
    main()
