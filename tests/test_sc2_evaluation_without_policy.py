"""CPU-only: real collector/validation methods, with model and transport doubles."""
from collections import defaultdict
from copy import deepcopy
from types import SimpleNamespace
import json
import sys
import tempfile
from pathlib import Path
import unittest

import numpy as np

from agent_system.sc2_episode import policy_transitions
from test_r6_natural_termination import ROOT, method

sys.path.insert(0, str(ROOT/'scripts'))
from run_record import finish_evaluation


class Batch:
    def __init__(self, batch=None, non_tensor_batch=None, meta_info=None):
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch or {}
        self.meta_info = meta_info or {}

    @classmethod
    def from_single_dict(cls, data):
        return cls(batch={k: v for k, v in data.items() if k in (
            'input_ids', 'attention_mask', 'position_ids')},
            non_tensor_batch={k: v for k, v in data.items() if k not in (
                'input_ids', 'attention_mask', 'position_ids')})

    def repeat(self, repeat_times, interleave):
        return self

    def pop(self, batch_keys, non_tensor_batch_keys):
        return Batch({k: self.batch.pop(k) for k in batch_keys},
                     {k: self.non_tensor_batch.pop(k) for k in non_tensor_batch_keys})


def transition(reward, applied, done, uid='game', source='sc2', active=True):
    return dict(rewards=reward, policy_action_applied=applied, environment_done=done,
                active_masks=active, is_tool_query=False, traj_uid=uid, data_source=source,
                prompts=[11], responses=[22])


class EvaluationWithoutPolicyTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = SimpleNamespace(decode=lambda ids, **kw: str(list(ids)),
                                         eos_token_id=0, pad_token_id=0)
        self.config = SimpleNamespace(
            trainer=SimpleNamespace(val_only=True),
            env=SimpleNamespace(env_name='SC2', rollout=SimpleNamespace(n=1)),
            algorithm=SimpleNamespace(filter_groups=SimpleNamespace(enable=False)),
            actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(
                val_kwargs=SimpleNamespace(n=1, do_sample=False))),
            reward_model=SimpleNamespace(enable=False))
        scope = dict(List=list, Dict=dict, np=np, DataProto=Batch, EnvironmentManagerBase=object,
                     policy_transitions=policy_transitions, collate_fn=lambda rows: rows)
        self.gather = method(ROOT/'agent_system/multi_turn_rollout/rollout_loop.py',
                             'TrajectoryCollector', 'gather_rollout_data', **scope)
        self.loop = method(ROOT/'agent_system/multi_turn_rollout/rollout_loop.py',
                           'TrajectoryCollector', 'multi_turn_loop', **scope)
        self.validate = method(ROOT/'verl/trainer/ppo/ray_trainer.py',
                               'RayPPOTrainer', '_validate', DataProto=Batch, np=np,
                               defaultdict=defaultdict)

    def collect(self, trajectories, rewards, wins, is_train=False, tools=None, val_only=None):
        self.config.trainer.val_only = not is_train if val_only is None else val_only
        original = deepcopy(trajectories)
        collector = SimpleNamespace(config=self.config, tokenizer=self.tokenizer)
        collector.vanilla_multi_turn_loop = lambda **kw: (
            trajectories, np.array(rewards), np.array([len(t) for t in trajectories]),
            {'success_rate': np.array(wins)},
            np.array([t[0]['traj_uid'] for t in trajectories]),
            np.array(tools or [0]*len(trajectories)))
        collector.gather_rollout_data = lambda **kw: self.gather(collector, **kw)
        result = self.loop(collector, Batch(), None, None, is_train=is_train)
        self.assertEqual(trajectories, original)
        return result

    def finish(self, outputs):
        calls = []
        pending = iter(outputs)
        def collect(**kwargs):
            self.assertFalse(kwargs['is_train'])
            return next(pending)
        trainer = SimpleNamespace(
            config=self.config, tokenizer=self.tokenizer, actor_rollout_wg=None, val_envs=None,
            traj_collector=SimpleNamespace(multi_turn_loop=collect),
            val_dataloader=[dict(input_ids=np.array([[1]]), attention_mask=np.array([[1]]),
                position_ids=np.array([[0]]), raw_prompt_ids=np.array([[1]]),
                data_source=np.array(['sc2'])) for _ in outputs],
            val_reward_fn=lambda *a, **kw: self.fail('Evaluation must not fabricate reward tokens'),
            _maybe_log_val_generations=lambda **kw: calls.append(kw))
        metrics = self.validate(trainer)
        self.assertTrue(all(np.isfinite(v) for v in metrics.values()))
        return metrics, calls

    def test_no_decision_victory_and_defeat_finish_without_policy_tensors(self):
        for reward, won in ((20,1),(-50,0)):
            with self.subTest(reward=reward):
                output = self.collect([[transition(reward,False,True)]], [reward], [won])
                self.assertIsNone(output.batch)
                self.assertEqual(output.non_tensor_batch, {})
                metrics, calls = self.finish([output])
                self.assertEqual(metrics['val/sc2/test_score'], reward)
                self.assertEqual(metrics['val/success_rate'], won)
                self.assertEqual(metrics['val/sc2/episode_count'], 1)
                self.assertEqual(metrics['val/sc2/no_policy_episode_count'], 1)
                self.assertEqual(metrics['val/sc2/policy_decision_count'], 0)
                self.assertEqual(calls[0]['outputs'], [])

    def test_mixed_games_count_once_not_once_per_answer(self):
        trajectories = [[transition(20,False,True,uid='empty')],
                        [transition(1,True,False),transition(2,True,False),
                         transition(-50,False,True),transition(-50,False,True,active=False)]]
        output = self.collect(trajectories,[20,-47],[1,0],tools=[0,2])
        metrics, calls = self.finish([output])
        self.assertEqual(metrics['val/sc2/test_score'], -13.5)
        self.assertEqual(metrics['val/success_rate'], .5)
        self.assertEqual(metrics['val/sc2/tool_call_count/mean'], 1)
        self.assertEqual(metrics['val/sc2/policy_decision_count'], 2)
        self.assertEqual(len(calls[0]['outputs']), 2)
        self.assertEqual(calls[0]['scores'], [-47,-47])

    def test_unequal_validation_batches_use_game_weighting(self):
        a = self.collect([[transition(20,False,True)]],[20],[1])
        b = self.collect([[transition(-50,True,True,uid=str(i))] for i in range(3)],
                         [-50]*3,[0]*3)
        metrics, _ = self.finish([a,b])
        self.assertEqual(metrics['val/success_rate'], .25)
        self.assertEqual(metrics['val/sc2/test_score'], -32.5)
        self.assertEqual(metrics['val/sc2/episode_count'], 4)

    def test_normal_single_game_keeps_whole_reward_and_real_answers(self):
        output = self.collect([[transition(1,True,False),transition(20,True,True)]],[21],[1])
        metrics, calls = self.finish([output])
        self.assertEqual(metrics['val/sc2/test_score'],21)
        self.assertEqual(metrics['val/sc2/no_policy_episode_count'],0)
        self.assertEqual(len(calls[0]['inputs']),2)
        self.assertEqual(len(calls[0]['outputs']),2)

    def test_reward_sources_remain_separate(self):
        output = self.collect([[transition(20,False,True,source='a')],
                               [transition(-50,True,True,source='b')]], [20,-50],[1,0])
        metrics, _ = self.finish([output])
        self.assertEqual(metrics['val/a/test_score'],20)
        self.assertEqual(metrics['val/b/test_score'],-50)

    def test_unconsumed_nonterminal_still_fails_in_evaluation(self):
        with self.assertRaisesRegex(RuntimeError,'Unconsumed nonterminal'):
            self.collect([[transition(1,False,False)]],[1],[0])

    def test_training_empty_policy_still_fails_before_collation(self):
        with self.assertRaisesRegex(RuntimeError,'No consumed policy'):
            self.collect([[transition(20,False,True)]],[20],[1],is_train=True)

    def test_training_mixed_batch_preserves_only_real_credit(self):
        # Replace collation with an identity boundary to inspect actual rows.
        self.gather.__globals__['DataProto'] = SimpleNamespace(from_single_dict=lambda data: data)
        output = self.collect([[transition(20,False,True,uid='empty')],
                               [transition(1,True,False),transition(-50,False,True)]],
                              [20,-49],[1,0],is_train=True)
        self.assertEqual(len(output),1)
        self.assertEqual(output[0]['rewards'],-49)
        self.assertEqual(output[0]['episode_rewards'],-49)
        self.assertTrue(output[0]['policy_action_applied'])
        self.assertEqual(output[0]['episode_lengths'],1)

    def test_non_sc2_and_training_validation_do_not_gain_an_empty_batch_bypass(self):
        for name, standalone in (('SC2',False),('Other',True)):
            with self.subTest(env=name, standalone=standalone):
                self.config.env.env_name = name
                with self.assertRaisesRegex(RuntimeError,'No consumed policy'):
                    self.collect([[transition(20,False,True)]],[20],[1],val_only=standalone)

    def test_no_games_is_not_a_successful_evaluation(self):
        with self.assertRaisesRegex(RuntimeError,'No SC2 evaluation episodes'):
            self.collect([],[],[])

    def test_metadata_only_result_passes_actual_dataproto_consistency_check(self):
        output = self.collect([[transition(20,False,True)]],[20],[1])
        check = method(ROOT/'verl/protocol.py','DataProto','check_consistency',np=np)
        length = method(ROOT/'verl/protocol.py','DataProto','__len__')
        check(output)
        self.assertEqual(length(output),0)
        self.assertEqual(output.meta_info['sc2_evaluation_episodes'][0]['policy_decisions'],0)

    def test_recorded_r1_outcome_closes_new_cpu_fixture_not_original_failed_run(self):
        record_path = ROOT/'experiments/records/20260908-garrison-wave-ownership-v1/r1-result.json'
        original = record_path.read_bytes()
        record = json.loads(original)
        outcome = record['result']['episode_outcomes']['outcomes'][0]
        self.assertEqual(record['result']['process_exit_code'],1)
        self.assertEqual(outcome['policy_decision_count'],0)
        output = self.collect([[transition(record['cumulative_environment_reward'],False,True)]],
                              [record['cumulative_environment_reward']],[1])
        metrics, _ = self.finish([output])
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder/'manifest.json').write_text(json.dumps({'episode_api':'gymnasium-v0.26'}))
            (folder/'effective_config.json').write_text(json.dumps({'overrides':{
                'data.val_batch_size':'1','+env.val_rollout_n':'1'}}))
            (folder/'console.log').write_text('step:0 - '+' - '.join(f'{k}:{v}' for k,v in metrics.items()))
            (folder/'episode_outcomes.jsonl').write_text(json.dumps(outcome)+'\n')
            (folder/'prompt_response.jsonl').write_text(json.dumps({
                'policy_action_applied':False,'semantic_action_evaluable':False})+'\n')
            result = finish_evaluation(folder,0)
            self.assertEqual(result['status'],'completed')
            self.assertEqual(result['training_updates'],0)
            self.assertEqual(result['evaluable_action_count'],0)
            self.assertEqual(result['episode_outcomes']['wins'],1)
            self.assertEqual(result['metrics']['val/sc2/test_score'],20)
            self.assertEqual(finish_evaluation(folder,1)['status'],'failed')
            truncated = dict(outcome,terminated=False,truncated=True,game_result=None,
                             termination_reason='interaction_limit')
            (folder/'episode_outcomes.jsonl').write_text(json.dumps(truncated)+'\n')
            self.assertEqual(finish_evaluation(folder,0)['status'],'failed')
        self.assertEqual(record_path.read_bytes(),original)


if __name__ == '__main__':
    unittest.main()
