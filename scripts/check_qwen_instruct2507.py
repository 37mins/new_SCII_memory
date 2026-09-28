#!/usr/bin/env python3
import argparse
import hashlib
import json

from transformers import AutoConfig, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the Qwen Instruct model interface before SC2 rollout.")
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    messages = [
        {"role": "system", "content": "Return only the requested build-order format."},
        {"role": "user", "content": "Choose the next StarCraft II build-order action."},
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if config.architectures != ["Qwen3ForCausalLM"]:
        raise RuntimeError(f"Unexpected architecture: {config.architectures}")
    if "<think>" in rendered or "</think>" in rendered:
        raise RuntimeError("Qwen3-4B-Instruct-2507 template emitted a thinking block")

    print(
        json.dumps(
            {
                "architecture": config.architectures[0],
                "model_type": config.model_type,
                "prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
                "prompt_tokens": len(tokenizer(rendered, add_special_tokens=False).input_ids),
                "thinking_markers": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
