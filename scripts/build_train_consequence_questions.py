import argparse
import csv
import json
from pathlib import Path


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


def build_questions(
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

            statement = row["statement"]
            consequence_prompt = conseq["consequence_prompt"]
            if consequence_prompt:
                consequence_prompt = consequence_prompt[0].lower() + consequence_prompt[1:]
            full_question = (
                f"{statement} According to the assumption above, {consequence_prompt}"
            )

            out.append(
                {
                    "id": row["id"],
                    "base_fact": base_fact,
                    "proposition_id": row["proposition_id"],
                    "stance_label": row["stance_label"],
                    "template_id": template_id,
                    "statement": statement,
                    "consequence_prompt": conseq["consequence_prompt"],
                    "expected_answer": conseq["expected_answer"],
                    "full_question": full_question,
                }
            )
    return out


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("No rows matched train facts/templates with consequences.")
    fieldnames = [
        "id",
        "base_fact",
        "proposition_id",
        "stance_label",
        "template_id",
        "statement",
        "consequence_prompt",
        "expected_answer",
        "full_question",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build consequence questions for train facts/templates and "
            "write a CSV with full questions."
        )
    )
    parser.add_argument("--splits", default="dataset/splits.json", help="Path to splits.json.")
    parser.add_argument("--data", default="dataset/data.csv", help="Path to data.csv.")
    parser.add_argument(
        "--consequences", default="dataset/consequences.csv", help="Path to consequences.csv."
    )
    parser.add_argument(
        "--out",
        default="dataset/train_consequence_questions.csv",
        help="Path to output CSV.",
    )
    args = parser.parse_args()

    train_facts, train_templates = load_splits(Path(args.splits))
    consequences = load_consequences(Path(args.consequences))
    rows = build_questions(Path(args.data), train_facts, train_templates, consequences)
    write_csv(Path(args.out), rows)

    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
