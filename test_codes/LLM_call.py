#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path("/home/yyq_ghf/code/new_SCII")
DEFAULT_MODEL_PATH = REPO_ROOT / "assets" / "Qwen3-4B-Instruct-2507"


def parse_args():
	parser = argparse.ArgumentParser(description="Ask the local Qwen3-4B-Instruct-2507 model questions.")
	parser.add_argument(
		"--model",
		default=os.environ.get("MODEL_PATH", str(DEFAULT_MODEL_PATH)),
		help="Local Hugging Face model directory (default: MODEL_PATH or the repository assets directory).",
	)
	parser.add_argument(
		"--device",
		default="cuda:0" if torch.cuda.is_available() else "cpu",
		help="Device used for inference, for example cuda:0 or cpu.",
	)
	parser.add_argument("--prompt", help="Ask one question and exit; omit for interactive chat.")
	parser.add_argument("--max-new-tokens", type=int, default=512)
	parser.add_argument("--temperature", type=float, default=0.7)
	parser.add_argument("--top-p", type=float, default=0.8)
	return parser.parse_args()


def select_dtype(device):
	if device.type == "cuda":
		return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
	return torch.float32


def generate_answer(model, tokenizer, messages, device, max_new_tokens, temperature, top_p):
	input_ids = tokenizer.apply_chat_template(
		messages,
		tokenize=True,
		add_generation_prompt=True,
		return_tensors="pt",
		enable_thinking=False,
	).to(device)
	attention_mask = torch.ones_like(input_ids)
	generation_args = {
		"input_ids": input_ids,
		"attention_mask": attention_mask,
		"max_new_tokens": max_new_tokens,
		"eos_token_id": tokenizer.eos_token_id,
		"pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
		"do_sample": temperature > 0,
	}
	if temperature > 0:
		generation_args["temperature"] = temperature
		generation_args["top_p"] = top_p

	with torch.inference_mode():
		output_ids = model.generate(**generation_args)
	answer_ids = output_ids[0, input_ids.shape[1]:]
	return tokenizer.decode(answer_ids, skip_special_tokens=True).strip()


def main():
	args = parse_args()
	model_path = Path(args.model).expanduser().resolve()
	if not model_path.is_dir():
		raise FileNotFoundError(f"Model directory does not exist: {model_path}")
	if args.max_new_tokens < 1:
		raise ValueError("--max-new-tokens must be greater than zero")

	device = torch.device(args.device)
	if device.type == "cuda" and not torch.cuda.is_available():
		raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")

	dtype = select_dtype(device)
	print(f"Loading model from {model_path} on {device} ({dtype})...", flush=True)
	tokenizer = AutoTokenizer.from_pretrained(
		model_path,
		local_files_only=True,
		trust_remote_code=False,
	)
	model = AutoModelForCausalLM.from_pretrained(
		model_path,
		torch_dtype=dtype,
		local_files_only=True,
		trust_remote_code=False,
	).to(device)
	model.eval()
	print("Model ready. Type /exit to quit.", flush=True)

	messages = []
	if args.prompt is not None:
		messages.append({"role": "user", "content": args.prompt})
		print(generate_answer(
			model, tokenizer, messages, device,
			args.max_new_tokens, args.temperature, args.top_p,
		))
		return

	while True:
		try:
			prompt = input("You: ").strip()
		except (EOFError, KeyboardInterrupt):
			print()
			break
		if prompt.lower() in {"/exit", "/quit", "exit", "quit"}:
			break
		if not prompt:
			continue

		messages.append({"role": "user", "content": prompt})
		answer = generate_answer(
			model, tokenizer, messages, device,
			args.max_new_tokens, args.temperature, args.top_p,
		)
		print(f"Qwen: {answer}\n")
		messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
	main()
