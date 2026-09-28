"""Original/A4/B4/A8: same feedback interface, eight declared game seeds."""
import json
import os
from pathlib import Path

import feedback_eval

RUNTIME_SHA = "361cb7ae7fd86fe8811d6fcb6946d0241bf93e11"
PREFIX = "20260911-execution-failure-feedback-v1-spb2-original-l6-g2-"
# Three known diagnostic seeds, followed by RandomState(20260911)'s first five
# randint(65536, 2**32-1). Declared before evaluation; do not replace after results.
GAME_SEEDS = [42613, 52620, 17089, 4018390961, 3236598124, 3297512220, 773351930, 1098341402]
# Reused intact from 2bf247d8 learning_eval.py; previously evaluated, not a blind panel.
SCENE_POST_GAME_SEEDS = [2259791793, 1436145665, 363527354, 462000319,
                         3434611143, 1968356657, 4276540916, 2065928601]
# Fixed before screening; see the September14 scene-screening record.
SCREEN_GAME_SEEDS = [3974528993, 1183530000, 3446496864, 2996859384,
                     3847406327, 1824802814, 175019443, 3122016241]


def checkpoint_preflight(runs):
    """Small files and shard lengths only; not proof of distributed loading."""
    proof = {}
    for arm in ("A", "B"):
        run = runs / (PREFIX + f"{arm}-rl4-4gpu-s0-r2")
        manifest = json.loads((run / "manifest.json").read_text())
        result = json.loads((run / "result.json").read_text())
        if (manifest["source_commit"] != RUNTIME_SHA or manifest["dirty"]
                or manifest["world_size"] != 4 or result != json.loads((run / ".done").read_text())
                or result["status"] != "completed" or result["process_exit_code"] != 0
                or result["completed_steps"] != 4 or not result["checkpoint_complete"]):
            raise ValueError(f"{arm}: completed frozen r2 step4 is required")
        checkpoint = run / "checkpoints/global_step_4"
        sizes = {}
        for rank in range(4):
            for kind, expected in (("model", 5555629511), ("optim", 10332463827), ("extra_state", None)):
                name = f"actor/{kind}_world_size_4_rank_{rank}.pt"
                size = (checkpoint / name).stat().st_size
                if (size != expected if expected else size <= 10000):
                    raise ValueError(f"Incomplete {arm} shard: {name}")
                sizes[name] = size
        if not (checkpoint / "data.pt").is_file():
            raise ValueError(f"{arm}: missing data.pt")
        proof[arm] = dict(run_id=run.name, checkpoint=str(checkpoint), files=sizes,
                          actual_distributed_load_still_required=True)
    return proof


def install():
    label = os.environ["SC2_FEEDBACK_LEARNING_MODEL"]
    if label not in ("original", "A4", "B4", "A8", "T08A8") or os.environ["SC2_EXECUTION_FAILURE_FEEDBACK"] != "1":
        raise ValueError("All learning-evaluation models must receive identical feedback")
    step = 0 if label == "original" else 8 if label in ("A8", "T08A8") else 4
    feedback_eval.install(expected_step=step, game_seeds=GAME_SEEDS)


def install_scene(game_seeds=None):
    """A4 versus its one-scene step5, feedback off in both evaluation arms."""
    from continue_a4 import SCENE_SEED
    label = os.environ["SC2_FEEDBACK_LEARNING_MODEL"]
    if label not in ("A4", "A5scene") or os.environ["SC2_EXECUTION_FAILURE_FEEDBACK"] != "0":
        raise ValueError("Scene evaluation requires A4/A5scene with frozen feedback off")
    feedback_eval.install(expected_step=4 if label == "A4" else 5,
                          game_seeds=[SCENE_SEED]*8 if game_seeds is None else game_seeds)


def install_multiseed():
    """Same A4/step5 interface, using the previously declared eight-seed panel."""
    install_scene(game_seeds=GAME_SEEDS)


def install_screen_scene():
    """One approved new scene, eight online A4 games, existing evaluation path."""
    index = int(os.environ["SC2_SCREEN_SCENE_INDEX"])
    if os.environ["SC2_FEEDBACK_LEARNING_MODEL"] != "A4" or not 0 <= index < len(SCREEN_GAME_SEEDS):
        raise ValueError("Screening requires A4 and one of the eight predeclared scenes")
    install_scene(game_seeds=[SCREEN_GAME_SEEDS[index]] * 8)


def install_scene_posteval():
    """Common held-out-from-this-training panel; only checkpoint differs."""
    label = os.environ['SC2_FEEDBACK_LEARNING_MODEL']
    if label not in ('A4', 'SceneA8', 'SceneB8') or os.environ['SC2_EXECUTION_FAILURE_FEEDBACK'] != '0':
        raise ValueError('Scene post-evaluation requires A4/SceneA8/SceneB8 with feedback off')
    feedback_eval.install(expected_step=4 if label == 'A4' else 8,
                          game_seeds=SCENE_POST_GAME_SEEDS)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    proof = checkpoint_preflight(args.runs)
    with args.output.open("x") as stream:
        json.dump(proof, stream, indent=2)
