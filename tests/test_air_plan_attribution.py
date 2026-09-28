import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / 'experiments/records/20260908-planner-air-options-v1/attribute_plans.py'
spec = importlib.util.spec_from_file_location('air_plan_attribution', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class AttributionTest(unittest.TestCase):
    def test_invalid_rule_air_is_not_model_selection(self):
        rows = [dict(policy_action_applied=True, semantic_action_evaluable=True,
                     semantic_action_valid=valid, llm_response=raw, step_index=i, reward=0,
                     prompt='The remaining BO list in this phase:\n- starport x 1\nUse only exact')
                for i, (valid, raw) in enumerate(((True, '- starport'), (False, '- siege tank')))]
        lines = ["BO ori: - starport, BO: ['starport']",
                 "BO ori: - siege tank, BO: ['marine', 'starport']"]
        result = module.attribute(rows, lines)
        self.assertEqual(result['plan_counts']['valid_model_plan'], {'starport': 1})
        self.assertEqual(result['plan_counts']['invalid_rule_processed_plan'], {'marine': 1, 'starport': 1})
        self.assertEqual(result['records'][1]['air_added_not_literal_in_raw'], {'starport': 1})

    def test_alignment_mismatch_fails(self):
        with self.assertRaises(AssertionError):
            module.attribute([], ["BO ori: - starport, BO: ['starport']"])


if __name__ == '__main__':
    unittest.main()
