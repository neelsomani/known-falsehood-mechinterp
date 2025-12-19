"""
Interactive terminal chat, matching the setup described in docs/AGENTS.md.

Usage:
  python scripts/llama_interactive.py --model meta-llama/Llama-3.1-70B-Instruct

Type your prompt and press Enter. Type `exit` or `quit` (or hit Ctrl+D/Ctrl+C) to leave the session.
"""

import argparse
import sys
from typing import List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with Llama from the terminal.")
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-70B-Instruct",
        help="Hugging Face model id. Use a chat-tuned variant if you have it locally.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate per turn.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Softmax temperature; set to 0 for greedy decoding.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling parameter.",
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
        help="Optional system prompt to prepend to the conversation.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype for model weights.",
    )
    return parser.parse_args()


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


def build_prompt(tokenizer, messages: List[Dict[str, str]]) -> str:
    # Llama chat models expose a chat template; this keeps formatting consistent with the README flow.
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def main() -> None:
    args = parse_args()
    tokenizer, model = load_model(args.model, args.dtype)

    print(f"Loaded {args.model}. Type your message and press Enter (Ctrl+C/D or 'exit' to quit).", flush=True)

    while True:
        try:
            user_input = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.", file=sys.stderr)
            break

        if user_input.lower() in {"exit", "quit"}:
            print("Session ended.")
            break
        if not user_input:
            continue

        messages: List[Dict[str, str]] = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": user_input})
        prompt = build_prompt(tokenizer, messages)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        do_sample = args.temperature > 0
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=do_sample,
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        assistant_reply = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        print(f"Llama> {assistant_reply}", flush=True)

        # Stateless: do not persist messages across turns.


if __name__ == "__main__":
    main()
