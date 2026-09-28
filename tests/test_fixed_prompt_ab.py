"""CPU-only prompt mutation, sampling schedule and frozen-contract checks."""
import ast
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/"experiments/records/20260909-bo-segment-protocol-v1/fixed_prompt_ab.py"
spec=importlib.util.spec_from_file_location("fixed_prompt_ab",PATH)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FixedPromptTest(unittest.TestCase):
    def test_change_is_reversible_and_retains_protocol(self):
        from test_original_bo_segment import pipeline
        p=pipeline("- marine x 15",True)["query"]
        q=module.candidate(p)
        self.assertEqual(q.replace(module.CLARIFICATION,module.MARKER),p)
        self.assertIn("approximately 15 entries",q)
        self.assertIn("no hard limit of 15 expanded actions",q)
        self.assertEqual(p.split("<answer>")[1],q.split("<answer>")[1])

    def test_validator_contract_does_not_change(self):
        from test_r6_format_reward import strict_validator
        p="The remaining BO list in this phase:\n- marine x 20\n"+module.MARKER
        with patch.dict(os.environ,SC2_ORIGINAL_BO_SEGMENT="1"):
            v=strict_validator()
            for answer in ("- marine x 20","- marine x 2\n- marine x 20","- tank", "- marine\n"*21):
                self.assertEqual(v(p,answer),v(module.candidate(p),answer))

    def test_rejects_unknown_prompt_shape(self):
        for prompt in ("unrelated",module.MARKER*2):
            with self.assertRaises(ValueError):module.candidate(prompt)

    def test_full_counts_not_only_first_failure(self):
        from test_r6_format_reward import strict_validator
        p="The remaining BO list in this phase:\n- marine x 20\n"+module.MARKER
        with patch.dict(os.environ,SC2_ORIGINAL_BO_SEGMENT="1"):
            r=module.score(p,"- marine x 21\n- marine x",strict_validator())
        self.assertFalse(r["valid"])
        self.assertEqual(r["over_quota"],{"marine":[21,20]})

    def test_schedule_keeps_41_per_arm_and_pairs_rank_seed(self):
        schedule={arm:[(min(start+rank, start if start+rank>=41 else start+rank),rank,start+rank>=41)
                    for start in range(0,41,4) for rank in range(4)] for arm in ("A","B")}
        self.assertEqual(schedule["A"],schedule["B"])
        for arm in schedule:
            self.assertEqual([i for i,r,p in schedule[arm] if not p],list(range(41)))
            self.assertEqual(sum(p for i,r,p in schedule[arm]),3)

    def test_entry_has_no_training_game_or_decoding_override(self):
        tree=ast.parse(PATH.read_text())
        names=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
        self.assertIn("worker.generate_sequences",names)
        self.assertIn("collector.preprocess_single_sample",names)
        self.assertFalse(set(names)&{"worker.update_actor","worker.actor_optimizer.step","make_envs","run_game"})
        self.assertNotIn("repetition_penalty",PATH.read_text())


if __name__=="__main__":unittest.main()
