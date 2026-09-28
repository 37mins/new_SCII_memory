"""Opt-in incremental SC2 combat-result reward; no action or prompt changes."""
import json
import math
import os


CATEGORIES = ('army', 'economy', 'technology')


def combat_values(score):
    """Symmetric engine categories: army, workers/economy, production/tech.

    Minerals and gas use the same unit of value. Do not infer lost units from
    production counts, visible enemies, command ACKs or remaining resources.
    """
    return tuple(sum(float(getattr(score, f'{kind}_{resource}_{category}'))
                     for resource in ('minerals', 'vespene') for category in CATEGORIES)
                 for kind in ('killed', 'lost'))


class CombatResultReward:
    def __init__(self, scale):
        if not math.isfinite(scale) or scale < 0:
            raise ValueError('Combat reward scale must be finite and nonnegative')
        self.scale = scale
        self.previous = None

    def consume(self, killed, lost, has_policy):
        current = (killed, lost)
        if any(not math.isfinite(v) or v < 0 for v in current):
            raise ValueError('Invalid SC2 combat counters')
        if self.previous is not None and any(a < b for a, b in zip(current, self.previous)):
            raise ValueError('SC2 combat counters decreased within an episode')
        before = self.previous
        self.previous = current
        # Fixed-prefix progress and a no-policy terminal never receive policy
        # credit. First real query establishes the baseline, not a synthetic act.
        delta = 0.0 if before is None or not has_policy else (
            killed - before[0] - (lost - before[1]))
        return self.scale * delta


def consume_combat_reward(planner, boundary):
    ledger = getattr(planner, 'combat_result_reward', None)
    if ledger is None:
        return 0.0
    score, game_loop = planner.combat_score()
    killed, lost = combat_values(score)
    policy_id = planner.last_consumed_policy_action_id
    reward = ledger.consume(killed, lost, policy_id is not None)
    print('Combat result reward: ' + json.dumps(dict(
        pid=os.getpid(), game_loop=int(game_loop), boundary=boundary,
        last_consumed_policy_action_id=policy_id, killed_value=killed,
        lost_value=lost, scale=ledger.scale, reward=reward)), flush=True)
    return reward
