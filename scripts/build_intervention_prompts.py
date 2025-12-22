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
    split: str,
) -> list[dict]:
    split_facts, split_templates = load_split(splits_path, split)
    consequences = load_consequences(consequences_path)
    rows = build_consequence_rows(data_path, split_facts, split_templates, consequences)
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


def paired_template_id(template_id: str) -> str | None:
    if template_id.startswith("T_TRUE_"):
        return template_id.replace("T_TRUE_", "T_FALSE_", 1)
    if template_id.startswith("T_FALSE_"):
        return template_id.replace("T_FALSE_", "T_TRUE_", 1)
    return None


def load_split(path: Path, split: str) -> tuple[set[str], set[str]]:
    with path.open(encoding="utf-8") as f:
        splits = json.load(f)
    if split == "all":
        fact_sets = splits["fact_splits"].values()
        template_sets = splits["template_splits"].values()
        facts = set().union(*map(set, fact_sets))
        templates = set().union(*map(set, template_sets))
        return facts, templates
    if split not in splits["fact_splits"] or split not in splits["template_splits"]:
        raise ValueError(f"Unknown split: {split}")
    return set(splits["fact_splits"][split]), set(splits["template_splits"][split])


def build_consequence_pairs(
    tokenizer,
    system: str,
    splits_path: Path,
    data_path: Path,
    consequences_path: Path,
    limit_total: int | None,
    seed: int,
    split: str,
) -> list[dict]:
    split_facts, split_templates = load_split(splits_path, split)
    consequences = load_consequences(consequences_path)
    rows = []
    with data_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["stance_label"] not in {"declared_true", "declared_false"}:
                continue
            base_fact = row["base_fact"]
            template_id = row["template_id"]
            if base_fact not in split_facts or template_id not in split_templates:
                continue
            rows.append(row)

    by_key = {}
    for row in rows:
        key = (row["base_fact"], row["proposition_id"], row["template_id"])
        by_key[key] = row

    pairs = {}
    for row in rows:
        template_id = row["template_id"]
        partner_template = paired_template_id(template_id)
        if not partner_template:
            continue
        key = (row["base_fact"], row["proposition_id"], partner_template)
        partner = by_key.get(key)
        if not partner:
            continue
        pair_ids = sorted([template_id, partner_template])
        pair_id = f"{row['base_fact']}::{row['proposition_id']}::{pair_ids[0]}::{pair_ids[1]}"
        pairs.setdefault(pair_id, []).append(row)

    pair_ids = list(pairs.keys())
    random.seed(seed)
    random.shuffle(pair_ids)
    if limit_total is not None:
        pair_ids = pair_ids[:limit_total]

    prompts = []
    for pair_id in pair_ids:
        for row in pairs[pair_id]:
            key = (row["base_fact"], row["proposition_id"], row["stance_label"])
            conseq = consequences.get(key)
            if not conseq:
                continue
            consequence_prompt = conseq["consequence_prompt"]
            if consequence_prompt:
                consequence_prompt = consequence_prompt[0].lower() + consequence_prompt[1:]
            full_question = (
                f"\"{row['statement']}\" According to the assumption above, {consequence_prompt}"
            )
            full_prompt = build_prompt(
                tokenizer,
                system,
                full_question,
                add_generation_prompt=True,
            )
            prompts.append(
                {
                    "pair_id": pair_id,
                    "stance_label": row["stance_label"],
                    "prompt": full_prompt,
                    "statement": row["statement"],
                    "label": conseq["expected_answer"],
                    "choices": [" A", " B"],
                    "base_fact": row["base_fact"],
                    "proposition_id": row["proposition_id"],
                    "template_id": row["template_id"],
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
    parser.add_argument(
        "--task",
        choices=["truth", "consequence", "consequence-pairs", "all"],
        default="all",
    )
    parser.add_argument("--facts", default="dataset/facts.csv")
    parser.add_argument("--splits", default="dataset/splits.json")
    parser.add_argument("--data", default="dataset/data.csv")
    parser.add_argument("--consequences", default="dataset/consequences.csv")
    parser.add_argument("--limit-truth-per-class", type=int, default=None)
    parser.add_argument("--limit-consequence", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--split",
        choices=["train", "test", "all"],
        default="train",
        help="Which split to use for consequence prompts/pairs.",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--out-prompts",
        type=Path,
        default=Path("dataset/intervention_prompts.jsonl"),
        help="Output for non-paired prompts (truth + consequence).",
    )
    parser.add_argument(
        "--out-pairs",
        type=Path,
        default=Path("dataset/intervention_pairs.jsonl"),
        help="Output for paired consequence prompts.",
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
                tokenizer=tokenizer,
                system=args.system,
                splits_path=Path(args.splits),
                data_path=Path(args.data),
                consequences_path=Path(args.consequences),
                limit_total=args.limit_consequence,
                seed=args.seed,
                split=args.split,
            )
        )
    pair_prompts = []
    if args.task in {"consequence-pairs", "all"}:
        pair_prompts.extend(
            build_consequence_pairs(
                tokenizer=tokenizer,
                system=args.system,
                splits_path=Path(args.splits),
                data_path=Path(args.data),
                consequences_path=Path(args.consequences),
                limit_total=args.limit_consequence,
                seed=args.seed,
                split=args.split,
            )
        )

    if args.task == "all":
        if not prompts and not pair_prompts:
            raise ValueError("No prompts generated.")
        args.out_prompts.parent.mkdir(parents=True, exist_ok=True)
        with args.out_prompts.open("w", encoding="utf-8") as f:
            for row in prompts:
                f.write(json.dumps(row) + "\n")
        args.out_pairs.parent.mkdir(parents=True, exist_ok=True)
        with args.out_pairs.open("w", encoding="utf-8") as f:
            for row in pair_prompts:
                f.write(json.dumps(row) + "\n")
        print(f"Wrote {len(prompts)} prompts to {args.out_prompts}")
        print(f"Wrote {len(pair_prompts)} pairs to {args.out_pairs}")
        return

    out_path = args.out
    if out_path is None:
        out_path = args.out_pairs if args.task == "consequence-pairs" else args.out_prompts

    selected = pair_prompts if args.task == "consequence-pairs" else prompts
    if not selected:
        raise ValueError("No prompts generated.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(selected)} prompts to {out_path}")


if __name__ == "__main__":
    main()
