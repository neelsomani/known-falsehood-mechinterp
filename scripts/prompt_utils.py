from __future__ import annotations

from typing import Iterable


def build_prompt(tokenizer, system: str | None, user: str, *, add_generation_prompt: bool) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def find_statement_span(prompt: str, statement: str) -> tuple[int, int]:
    start = prompt.rfind(statement)
    if start == -1:
        raise ValueError("Statement not found in prompt.")
    return start, start + len(statement)


def spans_to_token_index(offsets: Iterable[tuple[int, int]], span: tuple[int, int]) -> int | None:
    span_start, span_end = span
    last_idx = None
    for idx, (start, end) in enumerate(offsets):
        if start == end:
            continue
        if end > span_start and start < span_end:
            last_idx = idx
    return last_idx
