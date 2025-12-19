import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


WORD_RE = re.compile(r"-?\d+|[^\W\d_]+", flags=re.UNICODE)


def tokenize_statement(statement: str) -> list[tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in WORD_RE.finditer(statement)]


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


def find_first_part_index(tokens: list[tuple[str, int, int]], base_parts: list[str]) -> int | None:
    if not base_parts:
        return None
    first_part = base_parts[0]
    for idx, (token, _, _) in enumerate(tokens):
        if normalize_token(token) == first_part:
            return idx
    return None


def find_last_part_index(tokens: list[tuple[str, int, int]], base_parts: list[str]) -> int | None:
    last_idx = None
    for idx, (token, _, _) in enumerate(tokens):
        if normalize_token(token) in base_parts:
            last_idx = idx
    return last_idx


def build_prompt(tokenizer, system: str | None, user: str) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def load_splits(path: Path) -> tuple[set[str], set[str], set[str], set[str]]:
    with path.open(encoding="utf-8") as f:
        splits = json.load(f)
    train_facts = set(splits["fact_splits"]["train"])
    test_facts = set(splits["fact_splits"]["test"])
    train_templates = set(splits["template_splits"]["train"])
    test_templates = set(splits["template_splits"]["test"])
    return train_facts, test_facts, train_templates, test_templates


def spans_to_token_index(offsets: list[tuple[int, int]], span: tuple[int, int]) -> int | None:
    span_start, span_end = span
    last_idx = None
    for idx, (start, end) in enumerate(offsets):
        if start == end:
            continue
        if end > span_start and start < span_end:
            last_idx = idx
    return last_idx


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture residual stream activations.")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-70B-Instruct")
    parser.add_argument("--data", default="dataset/data.csv")
    parser.add_argument("--splits", default="dataset/splits.json")
    parser.add_argument("--split", choices=["train", "test", "all"], default="train")
    parser.add_argument("--system", default=None, help="Optional system prompt.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument(
        "--out",
        default=None,
        help="Output .pt path. Defaults to activations_{split}.pt in the dataset directory.",
    )
    parser.add_argument(
        "--mmap",
        default=None,
        help="Optional mmap path. Defaults to activations_{split}.mmap alongside --out.",
    )
    parser.add_argument("--keep-mmap", action="store_true", help="Keep mmap file after saving .pt.")
    args = parser.parse_args()

    torch_dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.eval()

    train_facts, test_facts, train_templates, test_templates = load_splits(Path(args.splits))

    rows = []
    last_base_fact = None
    last_first_match_idx = None
    with Path(args.data).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            base_fact = (row.get("base_fact") or "").strip()
            template_id = (row.get("template_id") or "").strip()
            if args.split == "train":
                if base_fact not in train_facts or template_id not in train_templates:
                    continue
            elif args.split == "test":
                if base_fact not in test_facts or template_id not in test_templates:
                    continue

            statement = (row.get("statement") or "").strip()
            tokens = tokenize_statement(statement)
            base_parts = normalize_base_parts(base_fact)
            first_idx = find_first_part_index(tokens, base_parts)
            entity_idx = first_idx
            if entity_idx is None and base_fact and base_fact == last_base_fact and last_first_match_idx is not None:
                if 0 <= last_first_match_idx < len(tokens):
                    entity_idx = last_first_match_idx
            if entity_idx is None:
                entity_idx = find_last_part_index(tokens, base_parts)

            last_base_fact = base_fact
            last_first_match_idx = first_idx

            if entity_idx is None or entity_idx >= len(tokens):
                entity_span = None
            else:
                _, start, end = tokens[entity_idx]
                entity_span = (start, end)

            rows.append(
                {
                    "row_index": row_idx,
                    "id": row.get("id", ""),
                    "base_fact": base_fact,
                    "statement": statement,
                    "entity_span": entity_span,
                }
            )

    if not rows:
        raise ValueError("No rows matched the requested split.")

    out_path = Path(args.out) if args.out else Path(args.data).with_name(f"activations_{args.split}.pt")
    mmap_path = (
        Path(args.mmap)
        if args.mmap
        else out_path.with_suffix(".mmap")
    )

    sample_prompt = build_prompt(tokenizer, args.system, rows[0]["statement"])
    sample_inputs = tokenizer(
        sample_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    with torch.no_grad():
        sample_out = model(**sample_inputs, output_hidden_states=True, return_dict=True)
    num_layers = len(sample_out.hidden_states) - 1
    d_model = sample_out.hidden_states[-1].shape[-1]

    acts = np.memmap(
        mmap_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(rows), num_layers, 2, d_model),
    )

    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start:start + args.batch_size]
        prompts = [build_prompt(tokenizer, args.system, r["statement"]) for r in batch_rows]
        prompt_starts = []
        statement_spans = []
        entity_spans = []
        for prompt, r in zip(prompts, batch_rows):
            statement = r["statement"]
            prompt_start = prompt.rfind(statement)
            if prompt_start == -1:
                raise ValueError(f"Statement not found in prompt for id={r['id']}")
            prompt_starts.append(prompt_start)
            statement_spans.append((prompt_start, prompt_start + len(statement)))
            if r["entity_span"] is None:
                entity_spans.append(None)
            else:
                e_start, e_end = r["entity_span"]
                entity_spans.append((prompt_start + e_start, prompt_start + e_end))

        tokenized = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets = tokenized.pop("offset_mapping")
        tokenized = {k: v.to(model.device) for k, v in tokenized.items()}

        with torch.no_grad():
            outputs = model(**tokenized, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states[1:]

        for i, r in enumerate(batch_rows):
            offsets_i = offsets[i].tolist()
            entity_span = entity_spans[i]
            statement_span = statement_spans[i]
            if entity_span is None:
                entity_tok = None
            else:
                entity_tok = spans_to_token_index(offsets_i, entity_span)
            final_tok = spans_to_token_index(offsets_i, statement_span)

            if entity_tok is None or final_tok is None:
                raise ValueError(f"Token index missing for id={r['id']}")

            for layer_idx, layer_h in enumerate(hidden_states):
                acts[start + i, layer_idx, 0, :] = (
                    layer_h[i, entity_tok, :].detach().to(torch.float16).cpu().numpy()
                )
                acts[start + i, layer_idx, 1, :] = (
                    layer_h[i, final_tok, :].detach().to(torch.float16).cpu().numpy()
                )

    acts.flush()

    tensor = torch.from_numpy(acts)
    payload = {
        "activations": tensor,
        "positions": ["entity", "final"],
        "ids": [r["id"] for r in rows],
        "data_row_indices": [r["row_index"] for r in rows],
        "split": args.split,
        "dtype": "float16",
    }
    torch.save(payload, out_path)

    if not args.keep_mmap:
        mmap_path.unlink(missing_ok=True)

    print(f"Wrote activations to {out_path}")


if __name__ == "__main__":
    main()
