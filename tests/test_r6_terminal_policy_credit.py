"""Exercise real transport/collector methods without SC2, Ray or GPU imports."""

import ast
import os
import socket
import time
import unittest
from types import SimpleNamespace

from agent_system import sc2_episode
from test_r6_natural_termination import ARENA, ROOT, method


def row(reward, applied=True, done=False, active=True, valid=True):
    return dict(rewards=reward, policy_action_applied=applied,
                environment_done=done, active_masks=active,
                is_action_valid=valid, traj_uid="game")


class TerminalPolicyCreditTest(unittest.TestCase):
    def filter(self, rows):
        return sc2_episode.policy_transitions(rows)

    def test_no_decision_victory_or_defeat_has_no_policy_sample(self):
        for reward in (20.0, -50.0):
            self.assertEqual(self.filter([row(reward, applied=False, done=True)]), [])

    def test_real_last_action_keeps_terminal_reward_even_if_invalid(self):
        rows = [row(0.5), row(20.5, done=True, valid=False)]
        self.assertEqual(self.filter(rows), rows)

    def test_delayed_terminal_reward_is_attached_once_to_last_real_action(self):
        rows = [row(0.5), row(20.0, applied=False, done=True),
                row(20.0, applied=False, done=True, active=False)]
        result = self.filter(rows)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["rewards"], 20.5)
        self.assertEqual(rows[0]["rewards"], 0.5)

    def test_unconsumed_nonterminal_is_not_silently_discarded(self):
        with self.assertRaisesRegex(RuntimeError, "nonterminal"):
            self.filter([row(0.5, applied=False)])

    def test_only_get_action_consumes_action_id_not_terminal_ack_loop(self):
        get_action = method(ARENA / "agents/planner.py", "SC2Planner", "get_action",
                            os=os, final_state_shaping_reward=lambda obs: 0)
        sent = []
        planner = SimpleNamespace(last_consumed_policy_action_id=None,
                                  correct_format=False, exceed=False, _send=sent.append,
                                  _recv=lambda: dict(msg="ACTION", value="- marine", policy_action_id=7))
        self.assertEqual(get_action(planner, "query"), "- marine")
        self.assertIsNone(sent[0]["last_consumed_policy_action_id"])
        self.assertEqual(planner.last_consumed_policy_action_id, 7)
        ack = method(ARENA / "agents/planner.py", "SC2Planner", "_wait_terminal_ack", time=time, socket=socket)
        replies = iter([dict(msg="ACTION", policy_action_id=8), dict(msg="TERMINAL_ACK")])
        planner._recv = lambda: next(replies)
        planner.terminal_ack_timeout = 1
        planner.client_socket = SimpleNamespace(gettimeout=lambda: None, settimeout=lambda timeout: None)
        ack(planner, {"last_consumed_policy_action_id": 7})
        self.assertEqual(planner.last_consumed_policy_action_id, 7)
        self.assertEqual(sent[-1]["last_consumed_policy_action_id"], 7)

    def test_reset_caches_and_acknowledges_initial_terminal_without_action(self):
        path = ROOT / "agent_system/environments/env_package/sc2_online_restored/envSC2.py"
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SC2Env")
        cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef)
                    and n.name in ("reset", "_mark_terminal_if_done")]
        class Base:
            def reset(self):
                pass
        event = SimpleNamespace(set=lambda: None, wait=lambda **kwargs: True)
        scope = dict(gym=SimpleNamespace(Env=Base), os=os, socket=socket,
                     TimeoutException=TimeoutError, get_unused_port=lambda: 12345,
                     mp=SimpleNamespace(Event=lambda: event, Process=lambda **kwargs: SimpleNamespace(start=lambda: None)))
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), scope)
        for result in ("Victory", "Defeat"):
            env = scope["SC2Env"]()
            terminal = dict(obs="terminal", done=True, last_consumed_policy_action_id=None,
                            **sc2_episode.outcome_fields(result))
            env.reset_retries = 1
            env._acquire_reset_slot = lambda: None
            env._arm_reset_alarm = lambda: False
            env._clear_reset_alarm = lambda armed: None
            env._release_reset_slot = lambda fd: None
            env._cleanup_resources = lambda: None
            env._setup_server = lambda port: None
            env._start_env_a_process = lambda: None
            env.accept_timeout = 1
            env.server_socket = SimpleNamespace(settimeout=lambda timeout: None, accept=lambda: (None, None))
            sent = []
            env._send_to_a = sent.append
            env._receive_from_a = lambda: terminal
            env._ack_terminal_to_a = lambda: sent.append(dict(msg="TERMINAL_ACK"))
            self.assertEqual(env.reset(), "terminal")
            self.assertTrue(env.game_end_flag)
            self.assertEqual(env.game_end_data, terminal)
            self.assertEqual(sent, [dict(msg="RESET"), dict(msg="TERMINAL_ACK")])

    def test_other_environments_keep_existing_active_mask_contract(self):
        sample = dict(rewards=1, active_masks=True, traj_uid="other")
        self.assertEqual(self.filter([sample]), [sample])

    def test_collector_carries_consumption_metadata_without_hiding_game_outcome(self):
        attach = method(
            ROOT / "agent_system/multi_turn_rollout/rollout_loop.py",
            "TrajectoryCollector", "attach_step_metadata", List=list, Dict=dict,
            np=SimpleNamespace(ndarray=list, array=lambda values, **kw: values),
            DataProto=object, torch_to_numpy=lambda values, **kw: values,
        )
        batch = SimpleNamespace(non_tensor_batch={})
        infos = [dict(is_action_valid=True, policy_action_applied=False, terminated=True, truncated=False),
                 dict(is_action_valid=False, policy_action_applied=True, terminated=True, truncated=False)]
        attach(None, batch, ["group"] * 2, ["game0", "game1"], infos, [20, -50], [True, True])
        self.assertEqual(batch.non_tensor_batch['active_masks'], [True, True])
        self.assertEqual(batch.non_tensor_batch['policy_action_applied'], [False, True])
        self.assertEqual(batch.non_tensor_batch['environment_done'], [True, True])

    def test_gather_filters_before_collation_and_keeps_all_game_successes(self):
        gather = method(
            ROOT / "agent_system/multi_turn_rollout/rollout_loop.py",
            "TrajectoryCollector", "gather_rollout_data",
            List=list, Dict=dict, np=SimpleNamespace(ndarray=list, float32=float, mean=lambda a: sum(a)/len(a)),
            DataProto=SimpleNamespace(from_single_dict=lambda data: data),
            collate_fn=lambda rows: rows,
            policy_transitions=getattr(sc2_episode, "policy_transitions", None),
        )
        no_decision = [row(20, applied=False, done=True)]
        played = [row(0.5), row(-50, applied=False, done=True)]
        result = gather(None, [no_decision, played], [20, -49.5], [1, 2],
                        {"success_rate": [1, 0]}, ["game", "game"], [0, 0])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["episode_rewards"], -49.5)
        self.assertEqual(result[0]["rewards"], -49.5)
        self.assertEqual(result[0]["episode_lengths"], 1)
        self.assertEqual(result[0]["success_rate"], 0.5)
        with self.assertRaisesRegex(RuntimeError, "No.*policy"):
            gather(None, [no_decision], [20], [1], {"success_rate": [1]}, ["game"], [0])

    def test_transport_reports_consumption_not_just_socket_success(self):
        step = method(
            ROOT / "agent_system/environments/env_package/sc2_online_restored/envSC2.py",
            "SC2Env", "step", gymnasium_transition=sc2_episode.gymnasium_transition,
        )
        transition = method(
            ROOT / "agent_system/environments/env_package/sc2_online_restored/envSC2.py",
            "SC2Env", "_policy_transition", gymnasium_transition=sc2_episode.gymnasium_transition,
        )
        for consumed, terminal, cached, pending in (
            (1, False, False, False), (1, True, False, False),
            (0, True, False, False), (0, True, True, False), (0, True, False, True),
        ):
            data = dict(obs="same text", reward=20 if terminal else .5, done=terminal,
                        success=terminal, attack_ct=0, last_consumed_policy_action_id=consumed or None)
            if terminal:
                data.update(sc2_episode.outcome_fields("Victory"))
            sent = []
            env = SimpleNamespace(
                reset_failed=False, game_end_flag=cached, game_end_data=data,
                _next_policy_action_id=0, pre_action_terminal_drain_timeout=0,
                _receive_pending_from_a=lambda timeout: data if pending else None,
                _send_to_a=sent.append, _receive_from_a=lambda: data,
                _mark_terminal_if_done=lambda data: None,
            )
            env._policy_transition = lambda *args, **kwargs: transition(env, *args, **kwargs)
            obs, reward, terminated, truncated, info = step(env, "- marine")
            self.assertEqual(info["policy_action_applied"], bool(consumed))
            self.assertEqual(terminated, terminal)
            self.assertEqual(reward, 20 if terminal else .5)
            self.assertEqual(len(sent), 0 if cached or pending else 1)


if __name__ == "__main__":
    unittest.main()
