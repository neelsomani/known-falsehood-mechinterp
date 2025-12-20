import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


WORD_RE = re.compile(r"-?\d+|[^\W\d_]+", flags=re.UNICODE)


def tokenize_statement(statement: str) -> list[tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in WORD_RE.finditer(statement)]


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


def capture_for_split(
    args,
    tokenizer,
    model,
    train_facts: set[str],
    test_facts: set[str],
    train_templates: set[str],
    test_templates: set[str],
    split: str,
) -> None:
    rows = []
    with Path(args.data).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            base_fact = (row.get("base_fact") or "").strip()
            template_id = (row.get("template_id") or "").strip()
            if split == "train":
                if base_fact not in train_facts or template_id not in train_templates:
                    continue
            elif split == "test":
                if base_fact not in test_facts or template_id not in test_templates:
                    continue
            statement = (row.get("statement") or "").strip()
            entity_index_raw = (row.get("entity_token_index") or "").strip()
            if entity_index_raw == "":
                raise ValueError(f"Missing entity_token_index for id={row.get('id')}")
            try:
                entity_idx = int(entity_index_raw)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid entity_token_index {entity_index_raw!r} for id={row.get('id')}"
                ) from exc
            tokens = tokenize_statement(statement)
            if entity_idx < 0 or entity_idx >= len(tokens):
                raise ValueError(
                    f"entity_token_index out of range for id={row.get('id')}"
                )
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
        raise ValueError(f"No rows matched the requested split: {split}.")

    out_path = Path(args.out) if args.out else Path(args.data).with_name(f"activations_{split}.pt")
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
        "split": split,
        "dtype": "float16",
    }
    torch.save(payload, out_path)

    if not args.keep_mmap:
        mmap_path.unlink(missing_ok=True)

    print(f"Wrote activations to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture residual stream activations.")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-70B-Instruct")
    parser.add_argument("--data", default="dataset/data.csv")
    parser.add_argument("--splits", default="dataset/splits.json")
    parser.add_argument("--split", choices=["train", "test", "both"], default="both")
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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.eval()

    train_facts, test_facts, train_templates, test_templates = load_splits(Path(args.splits))

    if args.split == "both":
        capture_for_split(
            args,
            tokenizer,
            model,
            train_facts,
            test_facts,
            train_templates,
            test_templates,
            "train",
        )
        capture_for_split(
            args,
            tokenizer,
            model,
            train_facts,
            test_facts,
            train_templates,
            test_templates,
            "test",
        )
    else:
        capture_for_split(
            args,
            tokenizer,
            model,
            train_facts,
            test_facts,
            train_templates,
            test_templates,
            args.split,
        )


if __name__ == "__main__":
    main()
