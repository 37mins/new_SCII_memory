"""CPU-only paired-config and frozen-panel checks; never launch a model/game."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT/'experiments/records/20260916-combat-result-reward-v1'
sys.path.insert(0, str(RECORD))
from failure_panel_probe import load_panel, summarize_panel


class CombatExperimentTest(unittest.TestCase):
    def test_panel_has_unchanged_queries_and_twentyfour_paired_seeds(self):
        panel, cases = load_panel(RECORD/'failure-panel.json')
        self.assertEqual(len(cases), 6)
        self.assertEqual([s for c in cases for s in c['sampling_seeds']], list(range(12000,12024)))
        for c in cases:
            self.assertEqual(c['prompt'], panel['cases'][c['index']]['prompt'])

    def test_summary_counts_plans_without_calling_invalid_output_legal(self):
        panel, cases = load_panel(RECORD/'failure-panel.json')
        rows = []
        for arm in ('original', 'A4'):
            for c in cases:
                for s in c['sampling_seeds']:
                    base = dict(arm=arm, case=c['index'], seed=s)
                    rows.append(dict(base, event='generation_start', input_token_sha256='same', cuda_rng_sha256=str(s)))
                    valid = s % 4 != 0
                    rows.append(dict(base, event='generation', valid=valid,
                        parsed_queue=['marine']*42 if valid else ['unavailable'],
                        raw_answer='</answer>', eos_seen=True, prompt_sha256=c['prompt_sha256']))
        report = summarize_panel(rows, panel)
        for context in report['contexts']:
            for arm in context['arms'].values():
                self.assertEqual(arm['answers'], 8)
                self.assertEqual(arm['valid'], 6)
                self.assertEqual(arm['legal_plan_kinds'], 1)
                self.assertEqual(arm['marine42'], 6)
                self.assertEqual(arm['sampling_distribution'][0]['fraction_of_all_answers'], .75)
        rows.pop()
        with self.assertRaises(AssertionError):
            summarize_panel(rows, panel)

    def test_real_dry_launchers_only_differ_in_declared_variable(self):
        sha = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
        with tempfile.TemporaryDirectory(prefix='sc2-combat-config-') as temporary:
            root = Path(temporary)
            configs, manifests = [], []
            for arm in ('A', 'B', 'C', 'D'):
                out = root/arm
                ray = '/dev/shm/combat_test_' + arm
                env = dict(os.environ, DRY_RUN='1', VERIFY_ASSETS='0', SOURCE_SHA=sha,
                    RUN_ID='dry-' + arm, OUTPUT_DIR=str(out), RAY_TMPDIR=ray,
                    MODEL_PATH=str(root), SC2PATH=str(root), PYTHON_BIN=sys.executable,
                    FEEDBACK_STORAGE_ROOT=str(root), FEEDBACK_PYTHON_OVERLAY=str(root/'python'),
                    FEEDBACK_INCLUDE_ROOT=str(root/'include'), FEEDBACK_RL_GPUS='4,5,6,7')
                subprocess.run(['bash',str(RECORD/'run_arm.sh'),arm], cwd=ROOT, env=env,
                               check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                config = json.loads((out/'effective_config.json').read_text())['overrides']
                configs.append({k: v.replace(str(out), '<RUN>').replace('dry-'+arm, '<ID>').replace(ray, '<RAY>')
                                for k,v in config.items()})
                manifests.append(json.loads((out/'manifest.json').read_text()))
            differences = {k for k in configs[0].keys() | configs[1].keys() if configs[0].get(k) != configs[1].get(k)}
            self.assertEqual(differences, {'+ray_init.runtime_env.env_vars.SC2_COMBAT_REWARD_SCALE'})
            differences = {k for k in configs[1].keys() | configs[2].keys() if configs[1].get(k) != configs[2].get(k)}
            self.assertEqual(differences, {'actor_rollout_ref.rollout.temperature'})
            self.assertEqual(configs[1]['actor_rollout_ref.rollout.temperature'], '0.4')
            self.assertEqual(configs[2]['actor_rollout_ref.rollout.temperature'], '0.8')
            self.assertEqual(configs[2]['+ray_init.runtime_env.env_vars.SC2_COMBAT_REWARD_SCALE'], "'0.001'")
            differences = {k for k in configs[2].keys() | configs[3].keys() if configs[2].get(k) != configs[3].get(k)}
            self.assertEqual(differences, {'algorithm.gamma'})
            self.assertEqual(configs[2]['algorithm.gamma'], '0.95')
            self.assertEqual(configs[3]['algorithm.gamma'], '1.0')
            for cfg, manifest in zip(configs, manifests):
                self.assertEqual(cfg['trainer.sc2_skip_zero_signal'], 'False')
                self.assertEqual(cfg['trainer.total_epochs'], '4')
                self.assertEqual(cfg['+ray_init.runtime_env.env_vars.SC2_EXECUTION_FAILURE_FEEDBACK'], "'0'")
                self.assertEqual(cfg['+ray_init.runtime_env.env_vars.SC2_ORIGINAL_BO_SEGMENT'], "'1'")
                self.assertEqual(cfg['data.max_response_length'], '192')
                self.assertEqual(manifest['world_size'], 4)
                self.assertEqual(manifest['start_step'], 4)
                self.assertEqual(manifest['target_steps'], 8)

    def test_temperature_candidate_needs_separate_authorization(self):
        env = dict(os.environ, DRY_RUN='0', SC2_COMBAT_REWARD_AUTHORIZED='1',
                   SC2_COMBAT_TEMPERATURE_AUTHORIZED='0')
        result = subprocess.run(['bash', str(RECORD/'run_arm.sh'), 'C'], cwd=ROOT, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertNotEqual(result.returncode, 0)

    def test_gamma_candidate_needs_separate_authorization(self):
        env = dict(os.environ, DRY_RUN='0', SC2_COMBAT_REWARD_AUTHORIZED='1',
                   SC2_COMBAT_TEMPERATURE_AUTHORIZED='1', SC2_COMBAT_GAMMA_AUTHORIZED='0')
        for entry in ('run_arm.sh', 'run_temperature_node47.sh'):
            result = subprocess.run(['bash', str(RECORD/entry), 'D'], cwd=ROOT, env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()
