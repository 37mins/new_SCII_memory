"""Opt-in A4 evaluation: same first two answers, only automatic first threshold differs."""
import functools
import hashlib
import json
import os
from pathlib import Path

import feedback_eval

FIXTURE = json.loads(Path(__file__).with_name('first-attack-inputs.json').read_text())
GAME_SEEDS = [FIXTURE['seed']] * 4


def set_threshold(player, threshold):
    before = dict(player.strategy_params)
    if threshold not in (20,40) or before['first_attack_threshold'] != 20 or before['subsequent_attack_threshold'] != 10:
        raise ValueError('First-attack comparison requires original 20/10 and candidate 20 or 40')
    player.strategy_params['first_attack_threshold'] = threshold
    return dict(before=before, after=dict(player.strategy_params))


def select_answer(call, prompt, generated):
    if call >= 2:
        return generated
    row = FIXTURE['rows'][call]
    actual = hashlib.sha256(prompt.encode()).hexdigest()
    if actual != row['prompt_sha256']:
        raise ValueError(f'First-attack historical prefix mismatch: call={call}, sha={actual}')
    return row['answer']


def install():
    threshold = int(os.environ['SC2_FIRST_ATTACK_THRESHOLD_AB'])
    if (threshold not in (20,40) or os.environ['SC2_FEEDBACK_LEARNING_MODEL'] != 'A4'
            or os.environ['SC2_EXECUTION_FAILURE_FEEDBACK'] != '1'):
        raise ValueError('First-attack evaluation requires A4, feedback on, threshold20/40')
    feedback_eval.install(expected_step=4, game_seeds=GAME_SEEDS,
        answer_description='First two answers: same historical replay, not model choices; third onward: online A4')
    from agent_system.environments.env_package.sc2_online_restored import envSC2
    original_init = envSC2.LLMPlayer.__init__

    @functools.wraps(original_init)
    def init(player, *args, **kwargs):
        original_init(player, *args, **kwargs)
        proof = set_threshold(player, threshold)
        proof.update(game_pid=os.getpid(), threshold=threshold)
        output = Path(os.environ['SC2_EVAL_SEEDS_LOG_PATH']).parent
        with (output/f'game-{os.getpid()}-first-attack.json').open('x') as stream:
            json.dump(proof, stream, indent=2)

    envSC2.LLMPlayer.__init__ = init
    import agent_system.environments as environments
    original_make = environments.make_envs

    @functools.wraps(original_make)
    def make_envs(config):
        checkpoint = str(config.trainer.resume_from_path)
        if (not config.trainer.val_only or config.trainer.n_gpus_per_node != 4
                or not checkpoint.endswith('A-rl4-4gpu-s0-r2/checkpoints/global_step_4')):
            raise ValueError('First-attack evaluation requires four-card A4 checkpoint')
        envs, val = original_make(config)
        output = Path(os.environ['SC2_EVAL_SEEDS_LOG_PATH']).parent
        original_step = val.step
        reached = set()

        def intervene(actions):
            forwarded, records = list(actions), []
            for env, answer in enumerate(actions):
                call = val._episode_step_by_env[env]
                if val._terminal_by_env[env] or call >= 2:
                    continue
                prompt = val._last_prompts[env]
                forwarded[env] = select_answer(call, prompt, answer)
                valid, reason = val._validate_current_bo_response(prompt, forwarded[env])
                if not valid:
                    raise ValueError(f'Historical answer rejected: env={env}, call={call}: {reason}')
                if call == 1:
                    reached.add(env)
                records.append(dict(env=env, threshold=threshold, call_index=call,
                    prompt_sha256=FIXTURE['rows'][call]['prompt_sha256'], raw_generated=answer,
                    forwarded=forwarded[env], source='historical_answer_replay_not_model_choice'))
            with (output/'interventions.jsonl').open('a') as stream:
                for row in records:
                    stream.write(json.dumps(row, ensure_ascii=False)+'\n')
            return original_step(forwarded)

        val.step = intervene
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer
        original_validate = RayPPOTrainer._validate

        @functools.wraps(original_validate)
        def validate(trainer):
            result = original_validate(trainer)
            outcomes = [json.loads(line) for line in (output/'episode_outcomes.jsonl').read_text().splitlines()]
            counts = {o['env_index']:o['policy_decision_count'] for o in outcomes}
            coverage = dict(threshold=threshold, subsequent_threshold=10, reached_second_answer=sorted(reached),
                all_reached=reached==set(range(4)), games=4, game_seeds=GAME_SEEDS,
                online_model_choices_by_env={str(i):counts[i]-2 for i in sorted(reached)},
                historical_answers_per_game=2, independent_scenarios=1, optimizer_updates=0)
            with (output/'intervention_coverage.json').open('x') as stream:
                json.dump(coverage, stream, indent=2)
            if not coverage['all_reached']:
                raise ValueError('Not all four games reached the second-answer boundary')
            return result

        RayPPOTrainer._validate = validate
        return envs, val

    environments.make_envs = make_envs
