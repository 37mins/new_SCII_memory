"""One four-game execution-order intervention; evaluation only, never PPO."""
from collections import Counter
import functools
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ARMS = ["A", "B", "A", "B"]
SEED = 17089
PROMPT_HASHES = [
    "95f4e1650fe0f587aba982557916cbb0e59d9bcbf1c10092fd40664578a5d424",
    "8d4d419e666c5f0e8f1996895386ac4d003fca13ea5dca84a6c5497006f51c59",
]
OPENING = "<answer>\n- supplydepot x 4\n- barracks x 4\n- marine x 24\n</answer>"
ORIGINAL = "<answer>\n- marine x 1\n- supplydepot x 1\n- barracks x 1\n- reactor barracks x 2\n- scv x 1\n- refinery x 1\n</answer>"


def move_refinery(answer):
    lines = answer.splitlines()
    lines.remove("- refinery x 1")
    lines.insert(lines.index("- reactor barracks x 2"), "- refinery x 1")
    return "\n".join(lines)


def select_answer(call_index, arm, prompt, generated):
    if call_index >= 2:
        return generated
    actual = hashlib.sha256(prompt.encode()).hexdigest()
    if actual != PROMPT_HASHES[call_index]:
        raise ValueError(f"Intervention input differs from historical boundary: {call_index} {actual}")
    if call_index == 0:
        return OPENING
    return ORIGINAL if arm == "A" else move_refinery(ORIGINAL)


class FourSeeds:
    def __init__(self):
        self.used = False

    def randint(self, low, high, size):
        if self.used or (low, high, size) != (2**16, 2**32 - 1, 4):
            raise ValueError("Only one four-game evaluation batch authorized")
        self.used = True
        return np.array([SEED] * 4, dtype=np.int64)


def seed_worker(_worker):
    import torch
    rank = torch.distributed.get_rank()
    torch.manual_seed(rank // 2)
    output = Path(os.environ["SC2_EVAL_SEEDS_LOG_PATH"]).parent
    proof = dict(rank=rank, seed=rank // 2,
                 cuda_rng_sha256=hashlib.sha256(bytes(torch.cuda.get_rng_state().tolist())).hexdigest())
    with (output / f"sampling-rank{rank}.json").open("x") as stream:
        json.dump(proof, stream)


def install_game_observers():
    from agent_system.environments.env_package.sc2_online_restored import envSC2
    original_start = envSC2.SC2Env._start_env_a_process
    original_init = envSC2.LLMPlayer.__init__
    original_step = envSC2.SC2Env.step

    @functools.wraps(original_start)
    def start(self, *args, **kwargs):
        # Isolate observation artifacts, not gameplay/RNG; prevent same-second replay overwrite.
        os.environ["SC2_RUNTIME_LOG_ROOT"] += f"/game-{os.getpid()}"
        return original_start(self, *args, **kwargs)

    @functools.wraps(original_init)
    def init(player, *args, **kwargs):
        original_init(player, *args, **kwargs)
        send = player.commander.planner._send
        path = Path(os.environ["SC2_EVAL_SEEDS_LOG_PATH"]).parent / f"game-{os.getpid()}-boundaries.jsonl"

        def capture(data):
            row = dict(game_pid=os.getpid(), time=player.time_formatted,
                       minerals=player.minerals, vespene=player.vespene,
                       units=dict(Counter(u.type_id.name for u in player.units)),
                       structures=[dict(tag=u.tag, name=u.type_id.name, progress=u.build_progress,
                                        add_on_tag=u.add_on_tag) for u in player.structures],
                       bo_index=player.current_bo_index, bo_queue=player.bo_data,
                       done=data["done"], prompt_sha256=hashlib.sha256(data["obs"].encode()).hexdigest())
            with path.open("a") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            return send(data)

        player.commander.planner._send = capture

    @functools.wraps(original_step)
    def step(self, action):
        obs, reward, terminated, truncated, info = original_step(self, action)
        info["order_diagnostic_game_pid"] = self.env_a_process.pid
        return obs, reward, terminated, truncated, info

    envSC2.SC2Env._start_env_a_process = start
    envSC2.LLMPlayer.__init__ = init
    envSC2.SC2Env.step = step


def install():
    import agent_system.environments as environments
    install_game_observers()
    original_make = environments.make_envs

    @functools.wraps(original_make)
    def make_envs(config):
        if not config.trainer.val_only or config.trainer.resume_mode != "resume_path":
            raise ValueError("Order intervention is checkpoint evaluation only")
        checkpoint = Path(config.trainer.resume_from_path)
        if checkpoint.name != "global_step_24" or config.trainer.get("sc2_checkpoint_resume", False):
            raise ValueError("Expected step24 and fresh evaluation environments")
        envs, val = original_make(config)
        raw = val.envs
        if raw.is_train or raw.num_processes != 4 or raw.reset_index != 0 or raw.seed != 1000:
            raise ValueError("Expected four fresh validation workers")
        raw.validation_rng = FourSeeds()
        output = Path(os.environ["SC2_EVAL_SEEDS_LOG_PATH"]).parent
        original_step = val.step

        def intervene(actions):
            forwarded = list(actions)
            rows = []
            for i, answer in enumerate(actions):
                call = val._episode_step_by_env[i]
                if val._terminal_by_env[i] or call >= 2:
                    continue
                forwarded[i] = select_answer(call, ARMS[i], val._last_prompts[i], answer)
                valid, reason = val._validate_current_bo_response(val._last_prompts[i], forwarded[i])
                if not valid:
                    raise ValueError(f"Frozen intervention rejected: {reason}")
                rows.append(dict(env=i, arm=ARMS[i], call_index=call,
                                 prompt_sha256=PROMPT_HASHES[call], raw_generated=answer,
                                 forwarded=forwarded[i], source="historical_answer_replay_not_model_choice"))
            with (output / "interventions.jsonl").open("a") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            result = original_step(forwarded)
            with (output / "game_worker_mapping.jsonl").open("a") as stream:
                for i, info in enumerate(result[-1]):
                    stream.write(json.dumps(dict(env=i, arm=ARMS[i],
                        game_pid=info["order_diagnostic_game_pid"])) + "\n")
            return result

        val.step = intervene
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer
        original_validate = RayPPOTrainer._validate

        def validate(trainer):
            if trainer.global_steps != 24:
                raise ValueError("Step24 checkpoint load did not return")
            trainer.actor_rollout_wg.execute_all_sync(
                "actor_rollout_execute_with_func_generator", seed_worker)
            proof = dict(checkpoint=str(checkpoint), loaded_global_step=24,
                         checkpoint_load_returned=True, val_only=True, games=4,
                         game_seeds=[SEED]*4, arms=ARMS, optimizer_updates=0,
                         first_two_answers="frozen intervention; not model decisions",
                         subsequent_answers="online SpB, unchanged temperature1.0/top_p0.9/192tokens")
            with (output / "checkpoint_evaluation.json").open("x") as stream:
                json.dump(proof, stream, indent=2)
            print("Refinery order evaluation: " + json.dumps(proof), flush=True)
            return original_validate(trainer)

        RayPPOTrainer._validate = validate
        return envs, val

    environments.make_envs = make_envs
