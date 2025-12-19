import argparse
import csv
import re
import sys
import unicodedata
from pathlib import Path


WORD_RE = re.compile(r"-?\d+|[^\W\d_]+", flags=re.UNICODE)


def tokenize_statement(statement: str) -> list[str]:
    return WORD_RE.findall(statement)


def normalize_token(token: str) -> str:
    decomposed = unicodedata.normalize("NFKD", token)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return stripped.lower()


def normalize_base_parts(base_fact: str) -> list[str]:
    if base_fact in {"boiling_point", "freezing_point"}:
        return ["water"]
    parts = []
    for part in base_fact.split("_"):
        if not part:
            continue
        if part.startswith("neg") and part[3:].isdigit():
            parts.append(f"-{part[3:]}")
        else:
            parts.append(part)
    return [normalize_token(part) for part in parts]


def find_first_part_index(tokens: list[str], base_parts: list[str]) -> tuple[str | None, int | None]:
    if not base_parts:
        return None, None
    first_part = base_parts[0]
    for idx, token in enumerate(tokens):
        if normalize_token(token) == first_part:
            return token, idx
    return None, None


def find_last_part_index(tokens: list[str], base_parts: list[str]) -> tuple[str | None, int | None]:
    last_idx = None
    last_token = None
    for idx, token in enumerate(tokens):
        if normalize_token(token) in base_parts:
            last_idx = idx
            last_token = token
    return last_token, last_idx


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the last-matching token index in each statement based on base_fact parts."
        )
    )
    parser.add_argument(
        "--data",
        default="dataset/data.csv",
        help="Path to data.csv with base_fact and statement columns.",
    )
    parser.add_argument(
        "--out",
        default="dataset/entity_token_indices.csv",
        help="Path to write the output CSV.",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    out_path = Path(args.out)

    rows_out = []
    missing = 0
    total = 0
    last_base_fact = None
    last_first_match_idx = None

    with data_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            base_fact = (row.get("base_fact") or "").strip()
            statement = (row.get("statement") or "").strip()
            tokens = tokenize_statement(statement)
            base_parts = normalize_base_parts(base_fact)
            token, idx = find_first_part_index(tokens, base_parts)
            first_match_idx = idx
            if idx is None and base_fact and base_fact == last_base_fact and last_first_match_idx is not None:
                if 0 <= last_first_match_idx < len(tokens):
                    idx = last_first_match_idx
                    token = tokens[idx]
            if idx is None:
                token, idx = find_last_part_index(tokens, base_parts)
            if idx is None:
                missing += 1
                print(
                    f"Warning: no base_fact token match for id={row.get('id')} "
                    f"base_fact={base_fact!r}",
                    file=sys.stderr,
                )
            last_base_fact = base_fact
            last_first_match_idx = first_match_idx
            rows_out.append(
                {
                    "id": row.get("id", ""),
                    "base_fact": base_fact,
                    "statement": statement,
                    "entity_token": token or "",
                    "entity_token_index": "" if idx is None else str(idx),
                }
            )

    fieldnames = [
        "id",
        "base_fact",
        "statement",
        "entity_token",
        "entity_token_index",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Wrote {len(rows_out)} rows to {out_path}")
    print(f"Missing matches: {missing}/{total}")


if __name__ == "__main__":
    main()
