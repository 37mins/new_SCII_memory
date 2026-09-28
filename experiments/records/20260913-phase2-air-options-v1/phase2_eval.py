"""Four paired A4 games per arm; only the opt-in second-stage options differ."""
import functools
import hashlib
import json
import os
from pathlib import Path

GAME_SEEDS = [42613, 17089, 3236598124, 4018390961]


def install():
    import feedback_eval
    arm = os.environ["SC2_PHASE2_AIR_AB"]
    if (arm not in ("A", "B") or os.environ["SC2_FEEDBACK_LEARNING_MODEL"] != "A4"
            or os.environ["SC2_EXECUTION_FAILURE_FEEDBACK"] != "0"):
        raise ValueError("Phase-two comparison requires A4, feedback off, arm A/B")
    feedback_eval.install(expected_step=4, game_seeds=GAME_SEEDS)
    # install_game_observers above imports envSC2 and its actual Commander class.
    from agents.commander import SC2Strategist
    if arm == "B":
        from phase2_air_options import install as install_candidate
        install_candidate()
    generate = SC2Strategist.generate_strategy

    @functools.wraps(generate)
    def recorded(strategist, obs):
        phase = generate(strategist, obs)
        proof = dict(arm=arm, game_pid=os.getpid(), phase_index=phase["_phase_index"],
            strategist_class=type(strategist).__module__, bo_todo=phase["bo todo"],
            instruction_sha256=hashlib.sha256(phase["instruction"].encode()).hexdigest(),
            attack_instruction_sha256=hashlib.sha256(phase["attack_instruction"].encode()).hexdigest(),
            observation_sha256=hashlib.sha256(obs.encode()).hexdigest(),
            persistent_template_sha256=hashlib.sha256(json.dumps(strategist.strategy, sort_keys=True).encode()).hexdigest())
        output = Path(os.environ["SC2_EVAL_SEEDS_LOG_PATH"]).parent
        with (output / f"game-{os.getpid()}-phase2-options.jsonl").open("a") as stream:
            stream.write(json.dumps(proof, ensure_ascii=False) + "\n")
        return phase

    SC2Strategist.generate_strategy = recorded
