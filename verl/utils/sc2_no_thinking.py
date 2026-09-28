"""Opt-in SpB explicit no-thinking prefix and ASCII tag decoding constraint.

The same prefix-dependent support mask is used by HF sampling and PPO/ref
teacher forcing. This does not otherwise change the existing top-p protocol.
Only the deployed Qwen2 fast ByteLevel tokenizer is supported.
"""
from functools import lru_cache
import hashlib
import json

FORBIDDEN = ("<think", "</think")
EMPTY_THINKING_PREFIX = "<think>\n\n</think>\n"


def non_thinking_prompt(rendered):
    if not rendered.endswith("<|im_start|>assistant\n"):
        raise ValueError("SC2 no-thinking requires the verified SpB assistant header")
    return rendered + EMPTY_THINKING_PREFIX


class NoThinkingConstraint:
    def __init__(self, pieces):
        # Special tokens skipped by response decoding must not break a partial
        # tag. ByteLevel non-ASCII fragments cannot produce ASCII tag characters.
        self.pieces = {int(i): text.lower() for i, text in pieces.items()}
        self.states = sorted({""} | {s[:n] for s in FORBIDDEN for n in range(1, len(s))}, key=len)
        self.blocked = {
            state: tuple(sorted(i for i, piece in self.pieces.items()
                                if any(tag in state + piece for tag in FORBIDDEN)))
            for state in self.states
        }

    def advance(self, state, token):
        text = state + self.pieces.get(int(token), "")
        if any(tag in text for tag in FORBIDDEN):
            raise ValueError("SC2 no-thinking response violates the generation constraint")
        return next(s for s in reversed(self.states) if text.endswith(s))

    def prefix_state(self, tokens):
        state = ""
        for token in tokens:
            state = self.advance(state, token)
        return state

    def mask_generation(self, scores, prefixes):
        for row, tokens in enumerate(prefixes):
            blocked = self.blocked[self.prefix_state(tokens)]
            if blocked:
                scores[row, list(blocked)] = float("-inf")
        return scores

    def mask_teacher(self, logits, responses):
        """Mask predictions BEFORE each label; padding/EOS use the same rules."""
        import torch
        positions = {s: [] for s in self.states}
        for row, tokens in enumerate(responses.detach().cpu().tolist()):
            state = ""
            for col, token in enumerate(tokens):
                positions[state].append((row, col))
                state = self.advance(state, token)
        # Finite minimum yields zero probability, while avoiding 0 * -inf in
        # the existing entropy implementation and its backward pass.
        for state, rows in positions.items():
            blocked = self.blocked[state]
            if not rows or not blocked:
                continue
            indices = torch.tensor(rows, device=logits.device)
            banned = torch.tensor(blocked, device=logits.device)
            logits[indices[:, 0, None], indices[:, 1, None], banned[None, :]] = torch.finfo(logits.dtype).min
        return logits

    def proof(self):
        encoded = json.dumps(self.blocked, sort_keys=True, separators=(",", ":")).encode()
        return {"mode": "sc2-no-thinking-v1", "forbidden_prefixes": list(FORBIDDEN),
                "support_mask_sha256": hashlib.sha256(encoded).hexdigest(),
                "blocked_counts": {s: len(v) for s, v in self.blocked.items()}}


@lru_cache(maxsize=2)
def load_constraint(path):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    if type(tokenizer).__name__ != "Qwen2TokenizerFast" or backend["decoder"]["type"] != "ByteLevel":
        raise ValueError("SC2 no-thinking supports only the verified Qwen2 fast ByteLevel tokenizer")
    if tokenizer.encode("<think>", add_special_tokens=False) != [151667] or tokenizer.encode("</think>", add_special_tokens=False) != [151668]:
        raise ValueError("Unexpected SpB thinking token mapping")
    special = set(tokenizer.all_special_ids)
    pieces = {i: "" if i in special else tokenizer.convert_tokens_to_string([token])
              for token, i in tokenizer.get_vocab().items()}
    constraint = NoThinkingConstraint(pieces)
    print("SC2_NO_THINKING " + json.dumps(constraint.proof(), sort_keys=True), flush=True)
    return constraint


class NoThinkingLogitsProcessor:
    def __init__(self, constraint, prompt_length):
        self.constraint = constraint
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores):
        prefixes = input_ids[:, self.prompt_length:].detach().cpu().tolist()
        return self.constraint.mask_generation(scores, prefixes)


def validate_config(config):
    section = getattr(config, "actor_rollout_ref", {})
    paths = [section.get(key, {}).get("no_thinking_tokenizer_path")
             for key in ("actor", "ref", "rollout")]
    enabled = getattr(config, "data", {}).get("sc2_no_thinking", False)
    if not enabled and not any(paths):
        return
    if not enabled or not all(paths) or len(set(paths)) != 1:
        raise ValueError("No-thinking prefix, actor, reference and rollout must be enabled together")
    if config.actor_rollout_ref.rollout.name != "hf":
        raise ValueError("SC2 no-thinking requires the verified HF rollout")
