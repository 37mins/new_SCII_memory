"""One recorded supply-block intervention; eight evaluation games, never PPO."""
import functools
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ARMS = ["A", "B"] * 4
SEED = 52620
FIXTURE = json.loads(Path(__file__).with_name("supply-order-inputs.json").read_text())


def move_depot(answer):
    lines = answer.splitlines()
    depot = next(line for line in lines if line.startswith("- supplydepot"))
    lines.remove(depot)
    lines.insert(next(i for i, line in enumerate(lines) if line.startswith("- ")), depot)
    return "\n".join(lines)


def select_answer(call_index, arm, prompt, generated, prefix_answers=3):
    if call_index >= 3:
        return generated
    row = FIXTURE["rows"][call_index]
    actual = hashlib.sha256(prompt.encode()).hexdigest()
    if actual != row["prompt_sha256"]:
        raise ValueError(f"Supply diagnostic differs from recorded input: call={call_index}, sha={actual}")
    if call_index >= prefix_answers:
        return generated
    if call_index == 2 and arm == "B":
        return move_depot(row["answer"])
    return row["answer"]


def trace_payloads(payloads, calls, terminals, prefix_answers=3):
    """Label transport after the existing validator/projection; never alter text."""
    marked = []
    for i, payload in enumerate(payloads):
        if terminals[i]:
            origin = "terminal_unused"
        elif calls[i] < prefix_answers:
            origin = "human_order_intervention" if calls[i] == 2 and ARMS[i] == "B" else "historical_replay"
        else:
            origin = "online_model_transport"
        marked.append(dict(payload, trace_origin=origin))
    return marked


class EightSeeds:
    def __init__(self):
        self.used = False

    def randint(self, low, high, size):
        if self.used or (low, high, size) != (2**16, 2**32 - 1, 8):
            raise ValueError("Only one eight-game supply diagnostic is authorized")
        self.used = True
        return np.array([SEED] * 8, dtype=np.int64)


def write_game_event(player, event, **details):
    path = Path(os.environ["SC2_EVAL_SEEDS_LOG_PATH"]).parent / f"game-{os.getpid()}-supply-events.jsonl"
    row = dict(event=event, game_pid=os.getpid(), time=player.time_formatted,
               supply_used=player.supply_used, supply_cap=player.supply_cap,
               supply_left=player.supply_left, minerals=player.minerals,
               policy_action_id=player.commander.planner.last_consumed_policy_action_id,
               **details)
    with path.open("a") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def observe_callback(original, event):
    @functools.wraps(original)
    async def callback(player, unit):
        result = await original(player, unit)
        write_game_event(player, event, tag=unit.tag, unit=unit.type_id.name,
                         progress=unit.build_progress, position=list(unit.position))
        return result
    return callback


def install_execution_observers():
    from agent_system.environments.env_package.sc2_online_restored import envSC2
    from refinery_order_eval import install_game_observers
    install_game_observers()
    cls = envSC2.LLMPlayer
    original_init = cls.__init__

    @functools.wraps(original_init)
    def init(player, *args, **kwargs):
        original_init(player, *args, **kwargs)
        original_renew = player.commander.renew_bo

        def renew(*args, **kwargs):
            queue = original_renew(*args, **kwargs)
            write_game_event(player, "executor_queue", queue=queue,
                             remaining=dict(player.commander.waiting_bo))
            return queue

        player.commander.renew_bo = renew

    cls.__init__ = init
    for name in ("on_building_construction_started", "on_building_construction_complete", "on_unit_created"):
        setattr(cls, name, observe_callback(getattr(cls, name), name))


def install(prefix_answers=3):
    if prefix_answers not in (2, 3):
        raise ValueError("Supply evaluation supports only the recorded two/three-answer prefix")
    arms = ARMS if prefix_answers == 3 else ["online"] * 8
    import agent_system.environments as environments
    install_execution_observers()
    original_make = environments.make_envs

    @functools.wraps(original_make)
    def make_envs(config):
        if not config.trainer.val_only or config.trainer.resume_mode != "disable":
            raise ValueError("Supply diagnostic requires original-model evaluation, no resume")
        envs, val = original_make(config)
        raw = val.envs
        if raw.is_train or raw.seed != 1000 or raw.num_processes != 8 or raw.reset_index != 0:
            raise ValueError("Expected eight fresh validation workers")
        raw.validation_rng = EightSeeds()
        output = Path(os.environ["SC2_EVAL_SEEDS_LOG_PATH"]).parent
        if os.getenv("SC2_COMMAND_TRACE_DIR"):
            original_raw_step = raw.step

            def traced_step(payloads):
                return original_raw_step(trace_payloads(
                    payloads, val._episode_step_by_env, val._terminal_by_env, prefix_answers))

            raw.step = traced_step
        original_step = val.step
        reached = set()

        def intervene(actions):
            forwarded = list(actions)
            rows = []
            for i, answer in enumerate(actions):
                call = val._episode_step_by_env[i]
                if val._terminal_by_env[i] or call >= 3:
                    continue
                prompt = val._last_prompts[i]
                forwarded[i] = select_answer(call, arms[i], prompt, answer, prefix_answers)
                if call < prefix_answers:
                    valid, reason = val._validate_current_bo_response(prompt, forwarded[i])
                    if not valid:
                        raise ValueError(f"Frozen supply intervention rejected: {reason}")
                if call == 2:
                    reached.add(i)
                rows.append(dict(env=i, arm=arms[i], call_index=call,
                                 prompt_sha256=FIXTURE["rows"][call]["prompt_sha256"],
                                 raw_generated=answer, forwarded=forwarded[i],
                                 source=("historical_prefix_or_order_intervention_not_model_choice"
                                         if call < prefix_answers else "online_unmodified")))
            with (output / "interventions.jsonl").open("a") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            result = original_step(forwarded)
            with (output / "game_worker_mapping.jsonl").open("a") as stream:
                for i, info in enumerate(result[-1]):
                    stream.write(json.dumps(dict(env=i, arm=arms[i],
                        game_pid=info["order_diagnostic_game_pid"])) + "\n")
            return result

        val.step = intervene
        # Delayed until TaskRunner; no CUDA probe in worker setup.
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer
        from refinery_order_eval import seed_worker
        original_validate = RayPPOTrainer._validate

        def validate(trainer):
            if trainer.global_steps != 0:
                raise ValueError("Expected original model at global step0")
            trainer.actor_rollout_wg.execute_all_sync("actor_rollout_execute_with_func_generator", seed_worker)
            proof = dict(val_only=True, resume_mode="disable", loaded_global_step=0,
                         optimizer_updates=0, model=str(config.actor_rollout_ref.model.path),
                         games=8, game_seeds=[SEED]*8, independent_scenarios=1, arms=arms,
                         sampling_seed_by_rank=[0, 0, 1, 1, 2, 2, 3, 3],
                         first_three_answers="diagnostic intervention, not model choices",
                         subsequent_answers="online original SpB, frozen generation config")
            if prefix_answers == 2:
                del proof["first_three_answers"]
                proof.update(first_two_answers="historical replay, not model choices",
                             online_from_call=3, intervention="none after the first two answers")
            with (output / "supply_evaluation.json").open("x") as stream:
                json.dump(proof, stream, indent=2)
            print("Supply order evaluation: " + json.dumps(proof), flush=True)
            result = original_validate(trainer)
            with (output / "intervention_coverage.json").open("x") as stream:
                json.dump(dict(reached=sorted(reached), expected=list(range(8)),
                               all_reached=len(reached) == 8), stream)
            return result

        RayPPOTrainer._validate = validate
        return envs, val

    environments.make_envs = make_envs
