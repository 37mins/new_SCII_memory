"""B4-only same-prefix evaluation: swap only the second recorded answer."""
import functools
import hashlib
import json
import os
from pathlib import Path

import feedback_eval

FIXTURE = json.loads(Path(__file__).with_name("second-plan-inputs.json").read_text())
SEED = FIXTURE["seed"]
ARMS = ["control_B_answer", "replace_with_A_answer"] * 4


def select_answer(call, arm, prompt, generated, *, marine_only=False):
    if call >= 2:
        return generated
    row = FIXTURE["rows"][call]
    actual = hashlib.sha256(prompt.encode()).hexdigest()
    if actual != row["prompt_sha256"]:
        raise ValueError(f"Historical prefix mismatch: call={call}, sha={actual}")
    if call == 0:
        return row["answer"]
    if marine_only and arm != ARMS[0]:
        return row["control_answer"].replace("- marine x 8\n", "- marine x 42\n", 1)
    return row["control_answer" if arm == ARMS[0] else "replacement_answer"]


def install(*, marine_only=False):
    # Existing seed selection, per-game logs and checkpoint-load proof; no new
    # game observers or sampling/production/reward changes.
    arms = [ARMS[0], "marine_42_only"]*4 if marine_only else ARMS
    description = ("historical B answer versus that answer with Marine 8->42 only"
                   if marine_only else "recorded B/A intervention")
    feedback_eval.install(expected_step=4, game_seeds=[SEED]*8,
        answer_description=f"First answer: common historical prefix; second: {description}; "
                           "both not model choices. From third answer: online B4.")
    import agent_system.environments as environments
    original_make = environments.make_envs

    @functools.wraps(original_make)
    def make_envs(config):
        checkpoint = str(config.trainer.resume_from_path)
        if (not config.trainer.val_only or config.trainer.n_gpus_per_node != 4
                or not checkpoint.endswith("B-rl4-4gpu-s0-r2/checkpoints/global_step_4")
                or os.environ["SC2_EXECUTION_FAILURE_FEEDBACK"] != "1"):
            raise ValueError("Second-plan intervention requires four-card B4 evaluation with feedback")
        envs, val = original_make(config)
        output = Path(os.environ["SC2_EVAL_SEEDS_LOG_PATH"]).parent
        original_step = val.step
        reached = set()

        def intervene(actions):
            forwarded = list(actions)
            records = []
            for env, answer in enumerate(actions):
                call = val._episode_step_by_env[env]
                if val._terminal_by_env[env] or call >= 2:
                    continue
                prompt = val._last_prompts[env]
                forwarded[env] = select_answer(call, arms[env], prompt, answer, marine_only=marine_only)
                valid, reason = val._validate_current_bo_response(prompt, forwarded[env])
                if not valid:
                    raise ValueError(f"Recorded answer rejected: env={env}, call={call}: {reason}")
                if call == 1:
                    reached.add(env)
                source = ("constructed_marine_count_intervention_not_model_choice"
                          if marine_only and call == 1 and env % 2 else
                          "historical_intervention_not_model_choice")
                records.append(dict(env=env, arm=arms[env], call_index=call,
                    prompt_sha256=FIXTURE["rows"][call]["prompt_sha256"], raw_generated=answer,
                    forwarded=forwarded[env], source=source))
            with (output / "interventions.jsonl").open("a") as stream:
                for row in records:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            return original_step(forwarded)

        val.step = intervene
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer
        original_validate = RayPPOTrainer._validate

        @functools.wraps(original_validate)
        def validate(trainer):
            result = original_validate(trainer)
            outcomes = [json.loads(line) for line in (output / "episode_outcomes.jsonl").read_text().splitlines()]
            counts = {o["env_index"]: o["policy_decision_count"] for o in outcomes}
            coverage = dict(arms=arms, reached_second_answer=sorted(reached), expected=list(range(8)),
                all_reached=reached == set(range(8)), intervention_answers_per_game=2,
                # Collector's consumed-policy counters include replayed answers.
                # Keep them unchanged and report autonomous choices separately.
                online_model_choices_by_env={str(i): counts[i]-2 for i in sorted(reached)},
                sampling_seed_by_rank=[0,1,2,3], sampling_reset_per_call=False,
                independent_game_scenarios=1, optimizer_updates=0)
            with (output / "intervention_coverage.json").open("x") as stream:
                json.dump(coverage, stream, indent=2)
            if not coverage["all_reached"]:
                raise ValueError("Not all games reached the declared second-answer boundary")
            return result

        RayPPOTrainer._validate = validate
        return envs, val

    environments.make_envs = make_envs


def install_marine_only():
    install(marine_only=True)
