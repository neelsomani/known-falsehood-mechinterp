import argparse
import csv
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt_utils import build_prompt


def load_model(model_id: str, dtype: str):
    torch_dtype = getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    return tokenizer, model


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    do_sample = False
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def parse_truth_label(text: str) -> str | None:
    match = re.search(r"\b(true|false|unknown)\b", text.strip().lower())
    return match.group(1).capitalize() if match else None


def parse_ab_label(text: str) -> str | None:
    match = re.search(r"\b([ab])\b", text.strip(), flags=re.IGNORECASE)
    return match.group(1).upper() if match else None


def load_splits(path: Path) -> tuple[set[str], set[str]]:
    with path.open(encoding="utf-8") as f:
        splits = json.load(f)
    train_facts = set(splits["fact_splits"]["train"])
    train_templates = set(splits["template_splits"]["train"])
    return train_facts, train_templates


def load_consequences(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    conseq = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["base_fact"], row["proposition_id"], row["stance_label"])
            conseq[key] = row
    return conseq


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
                    "proposition_id": row["proposition_id"],
                    "stance_label": row["stance_label"],
                    "template_id": row["template_id"],
                    "statement": row["statement"],
                    "full_question": full_question,
                    "expected_answer": conseq["expected_answer"],
                }
            )
    return out


def run_truth_task(
    model,
    tokenizer,
    system: str | None,
    facts_path: Path,
    max_new_tokens: int,
    limit_per_class: int | None,
    show_prompts: bool,
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
                response = generate(
                    model,
                    tokenizer,
                    full_prompt,
                    max_new_tokens=max_new_tokens,
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

    print("Truth/False/Unknown task:")
    print(f"  True accuracy: {true_acc:.3f} ({true_correct}/{true_total})")
    print(f"  False accuracy: {false_acc:.3f} ({false_correct}/{false_total})")
    print(f"  Overall accuracy: {overall_acc:.3f} ({true_correct + false_correct}/{total})")
    print(f"  Unknown percent: {unknown_pct:.2f}% ({unknown_count}/{total})")
    return {
        "true_acc": true_acc,
        "false_acc": false_acc,
        "overall_acc": overall_acc,
        "unknown_pct": unknown_pct,
    }


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
) -> dict[str, float]:
    train_facts, train_templates = load_splits(splits_path)
    consequences = load_consequences(consequences_path)
    rows = build_consequence_rows(data_path, train_facts, train_templates, consequences)

    total = 0
    correct = 0
    unparsed = 0
    breakdown: dict[tuple[str, str, str], dict[str, int]] = {}

    for row in rows:
        if limit_total is not None and total >= limit_total:
            break
        full_prompt = build_prompt(
            tokenizer,
            system,
            row["full_question"],
            add_generation_prompt=True,
        )
        response = generate(
            model,
            tokenizer,
            full_prompt,
            max_new_tokens=max_new_tokens,
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

    print("Consequence task (train facts/templates):")
    print(f"  Accuracy: {acc:.3f} ({correct}/{total})")
    print(f"  Unparsed percent: {unparsed_pct:.2f}% ({unparsed}/{total})")
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
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run truth and consequence behavioral checks.")
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-70B-Instruct",
        help="Hugging Face model id.",
    )
    parser.add_argument(
        "--system",
        default=(
            "You are a helpful agent. By default, evaluate statements "
            "using your general world knowledge. When a statement is presented as an explicit "
            "assumption for the sake of reasoning, condition your reasoning on that assumption "
            "rather than on world knowledge. Do not treat statements as assumptions unless the "
            "user's phrasing indicates they are to be taken as such. Answer the user's prompt "
            "in 1 character (A/B)."
        ),
        help="Optional system prompt.",
    )
    parser.add_argument(
        "--task",
        choices=["truth", "consequence", "all"],
        default="all",
        help="Deprecated; both tasks always run and are summarized at the end.",
    )
    parser.add_argument("--facts", default="dataset/facts.csv", help="Path to facts.csv.")
    parser.add_argument("--splits", default="dataset/splits.json", help="Path to splits.json.")
    parser.add_argument("--data", default="dataset/data.csv", help="Path to data.csv.")
    parser.add_argument(
        "--consequences", default="dataset/consequences.csv", help="Path to consequences.csv."
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--limit-truth-per-class",
        type=int,
        default=None,
        help="Limit the number of true and false facts evaluated.",
    )
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
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype for model weights.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer, model = load_model(args.model, args.dtype)

    truth_metrics = run_truth_task(
        model,
        tokenizer,
        args.system,
        Path(args.facts),
        args.max_new_tokens,
        args.limit_truth_per_class,
        args.show_prompts,
    )
    print()

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
    )
    print()

    print("Final summary:")
    print(f"  Truth overall accuracy: {truth_metrics['overall_acc']:.3f}")
    print(f"  Truth unknown percent: {truth_metrics['unknown_pct']:.2f}%")
    print(f"  Consequence accuracy: {consequence_metrics['acc']:.3f}")
    print(f"  Consequence unparsed percent: {consequence_metrics['unparsed_pct']:.2f}%")


if __name__ == "__main__":
    main()
