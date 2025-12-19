import argparse
import csv
import json
import random
from pathlib import Path

TEMPLATES = {
    "T_TRUE_1": ("declared_true", "It is true that: {X}"),
    "T_TRUE_2": ("declared_true", "A correct statement: {X}."),
    "T_TRUE_3": ("declared_true", "Someone said '{X}', and that is correct."),
    "T_FALSE_1": ("declared_false", "It is false that: {X}"),
    "T_FALSE_2": ("declared_false", "A common myth: {X}"),
    "T_FALSE_3": ("declared_false", "Someone said '{X}', but that is incorrect."),
    "T_BARE": ("bare", "{X}"),
}

DECLARED_TEMPLATE_IDS = [tid for tid in TEMPLATES.keys() if tid != "T_BARE"]

INPUT_FIELDS_REQUIRED = [
    "True Fact",
    "False Fact",
    "Category",
    "ID",
]


def _assert_ab(label: str, val: str, where: str) -> None:
    if val == "":
        return
    if val not in ("A", "B"):
        raise ValueError(f"{where}: {label} must be A/B or empty, got {val!r}")


def convert(input_csv: Path, out_data_csv: Path, out_conseq_csv: Path, include_bare: bool) -> list[str]:
    rows_out = []
    conseq_out = []
    base_facts = []

    with input_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for req in INPUT_FIELDS_REQUIRED:
            if req not in reader.fieldnames:
                raise ValueError(f"Missing required column: {req!r}")

        for r in reader:
            base_fact = (r.get("ID") or "").strip()
            category = (r.get("Category") or "").strip()
            x_true = (r.get("True Fact") or "").strip()
            x_false = (r.get("False Fact") or "").strip()

            if not base_fact:
                raise ValueError("Found empty ID/base_fact")
            if not category:
                raise ValueError(f"{base_fact}: empty Category")
            if not x_true or not x_false:
                raise ValueError(f"{base_fact}: empty True Fact or False Fact")

            base_facts.append(base_fact)

            prop_true = f"{base_fact}__true"
            prop_false = f"{base_fact}__false"

            for template_id in DECLARED_TEMPLATE_IDS:
                stance_label, tmpl = TEMPLATES[template_id]

                row_id_true = f"{base_fact}__true__{template_id}"
                rows_out.append(
                    {
                        "id": row_id_true,
                        "base_fact": base_fact,
                        "proposition_id": prop_true,
                        "statement": tmpl.format(X=x_true),
                        "stance_label": stance_label,
                        "template_id": template_id,
                        "category": category,
                    }
                )

                row_id_false = f"{base_fact}__false__{template_id}"
                rows_out.append(
                    {
                        "id": row_id_false,
                        "base_fact": base_fact,
                        "proposition_id": prop_false,
                        "statement": tmpl.format(X=x_false),
                        "stance_label": stance_label,
                        "template_id": template_id,
                        "category": category,
                    }
                )

            if include_bare:
                stance_label, tmpl = TEMPLATES["T_BARE"]
                row_id_bare = f"{base_fact}__false__T_BARE"
                rows_out.append(
                    {
                        "id": row_id_bare,
                        "base_fact": base_fact,
                        "proposition_id": prop_false,
                        "statement": tmpl.format(X=x_false),
                        "stance_label": stance_label,
                        "template_id": "T_BARE",
                        "category": category,
                    }
                )

            tf_df_conseq = (r.get("True Fact Declared False Consequence") or "").strip()
            tf_df_ans = (r.get("Expected True Fact Declared False Answer") or "").strip()
            ff_dt_conseq = (r.get("False Fact Declared True Consequence") or "").strip()
            ff_dt_ans = (r.get("Expected False Fact Declared True Answer") or "").strip()

            _assert_ab("Expected True Fact Declared False Answer", tf_df_ans, base_fact)
            _assert_ab("Expected False Fact Declared True Answer", ff_dt_ans, base_fact)

            if tf_df_conseq:
                conseq_out.append(
                    {
                        "base_fact": base_fact,
                        "proposition_id": prop_true,
                        "stance_label": "declared_false",
                        "consequence_prompt": tf_df_conseq,
                        "expected_answer": tf_df_ans,
                    }
                )
            if ff_dt_conseq:
                conseq_out.append(
                    {
                        "base_fact": base_fact,
                        "proposition_id": prop_false,
                        "stance_label": "declared_true",
                        "consequence_prompt": ff_dt_conseq,
                        "expected_answer": ff_dt_ans,
                    }
                )

    ids = [r["id"] for r in rows_out]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate ids detected in output data.csv")

    data_fields = [
        "id",
        "base_fact",
        "proposition_id",
        "statement",
        "stance_label",
        "template_id",
        "category",
    ]
    with out_data_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=data_fields)
        w.writeheader()
        w.writerows(rows_out)

    conseq_fields = [
        "base_fact",
        "proposition_id",
        "stance_label",
        "consequence_prompt",
        "expected_answer",
    ]
    with out_conseq_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=conseq_fields)
        w.writeheader()
        w.writerows(conseq_out)

    return base_facts


def build_splits(out_splits_json: Path, base_facts: list[str], seed: int, include_bare: bool) -> None:
    base_facts = sorted(set(base_facts))
    rng = random.Random(seed)
    rng.shuffle(base_facts)

    n = len(base_facts)
    n_train = int(0.8 * n)
    fact_train = base_facts[:n_train]
    fact_test = base_facts[n_train:]

    template_pairs = {
        "1": ["T_TRUE_1", "T_FALSE_1"],
        "2": ["T_TRUE_2", "T_FALSE_2"],
        "3": ["T_TRUE_3", "T_FALSE_3"],
    }
    template_train = template_pairs["1"] + template_pairs["2"]
    template_test = template_pairs["3"]

    splits = {
        "seed": seed,
        "fact_splits": {
            "train": fact_train,
            "test": fact_test,
        },
        "template_splits": {
            "train": template_train,
            "test": template_test,
        },
        "template_pairs": template_pairs,
        "notes": {
            "fact_unit": "base_fact",
            "template_unit": "paired TRUE/FALSE templates",
            "bare_template_included": include_bare,
        },
    }

    with out_splits_json.open("w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2, ensure_ascii=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert wide facts CSV to long format.")
    parser.add_argument(
        "--input",
        default="dataset/facts.csv",
        help="Path to the wide input CSV.",
    )
    parser.add_argument(
        "--out-data",
        default="dataset/data.csv",
        help="Path to write the long-format data CSV.",
    )
    parser.add_argument(
        "--out-consequences",
        default="dataset/consequences.csv",
        help="Path to write the consequences CSV.",
    )
    parser.add_argument(
        "--out-splits",
        default="dataset/splits.json",
        help="Path to write the splits JSON.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for the fact split.",
    )
    parser.add_argument(
        "--no-bare",
        action="store_false",
        dest="include_bare",
        help="Disable bare template rows for false propositions.",
    )
    args = parser.parse_args()

    input_csv = Path(args.input)
    out_data = Path(args.out_data)
    out_conseq = Path(args.out_consequences)
    out_splits = Path(args.out_splits)

    base_facts = convert(input_csv, out_data, out_conseq, include_bare=args.include_bare)
    build_splits(out_splits, base_facts, seed=args.seed, include_bare=args.include_bare)

    n_bf = len(set(base_facts))
    total_rows = n_bf * (13 if args.include_bare else 12)
    print(f"Unique base facts: {n_bf}")
    print(f"Expected data rows: {total_rows}")


if __name__ == "__main__":
    main()
