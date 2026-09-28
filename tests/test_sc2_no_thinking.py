"""CPU-only string/state and configuration regressions; no heavy imports."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sc2_no_thinking", ROOT/"verl/utils/sc2_no_thinking.py")
nt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nt)


class Options(dict):
    __getattr__ = dict.__getitem__


class NoThinkingTest(unittest.TestCase):
    def setUp(self):
        self.pieces = {0: "", 1: "<think>", 2: "</think>", 3: "<", 4: "/",
                       5: "thi", 6: "nk", 7: ">", 8: "<answer>", 9: "marine",
                       10: "\n<THINK>", 11: "abc</thi", 12: "ink", 13: "<th"}
        self.c = nt.NoThinkingConstraint(self.pieces)

    def test_single_token_and_embedded_case_variants_blocked(self):
        self.assertTrue({1, 2, 10}.issubset(self.c.blocked[""]))
        for i in (1, 2, 10):
            with self.assertRaises(ValueError): self.c.prefix_state([i])

    def test_alternative_spelling_and_skipped_specials_cannot_bypass(self):
        for tokens in ([3, 5], [3, 4, 5], [11], [3, 5, 0]):
            state = self.c.prefix_state(tokens)
            self.assertIn(6, self.c.blocked[state])
            with self.assertRaises(ValueError): self.c.prefix_state([*tokens, 6])
        self.assertIn(12, self.c.blocked[self.c.prefix_state([13])])

    def test_legal_answer_is_preserved_and_unrelated_text_clears_prefix(self):
        self.assertEqual(self.c.prefix_state([8, 9]), "")
        self.assertEqual(self.c.prefix_state([3, 9]), "")
        self.assertNotIn(9, self.c.blocked[""])

    def test_hash_is_independent_of_vocab_iteration_order(self):
        self.assertEqual(self.c.proof(), nt.NoThinkingConstraint(dict(reversed(list(self.pieces.items())))).proof())

    def test_empty_thinking_is_prompt_only_and_header_mismatch_fails(self):
        p="<|im_start|>user\nquery<|im_end|>\n<|im_start|>assistant\n"
        self.assertEqual(nt.non_thinking_prompt(p), p+"<think>\n\n</think>\n")
        with self.assertRaises(ValueError): nt.non_thinking_prompt("unknown template")

    def test_config_default_off_and_all_four_switches_must_match(self):
        def cfg(enabled=False, paths=(None, None, None), backend="hf"):
            return NS(data=Options(sc2_no_thinking=enabled), actor_rollout_ref=Options(
                {k: Options(no_thinking_tokenizer_path=p, name=backend)
                 for k,p in zip(("actor", "ref", "rollout"), paths)}))
        nt.validate_config(cfg())
        nt.validate_config(cfg(True, ("/model",)*3))
        for c in (cfg(True), cfg(False, ("/model",)*3), cfg(True,("/model",None,"/model")),
                  cfg(True,("/a","/b","/a")), cfg(True,("/a",)*3,"vllm")):
            with self.assertRaises(ValueError): nt.validate_config(c)


if __name__ == "__main__": unittest.main()
