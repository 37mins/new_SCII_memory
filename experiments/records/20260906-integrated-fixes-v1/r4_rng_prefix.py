"""One CPU-only fixed-BO prefix via the real SC2Env child entry; no ACTION sent."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]


def record_first_planner(player, output):
    """Save the full in-memory trace at the request boundary, then send unchanged."""
    planner = player.commander.planner
    send = planner._send

    def capture(data):
        if not data['done']:
            with (output / 'boundary_trace.json').open('x') as stream:
                json.dump(player.trace, stream, ensure_ascii=False)
            with (output / 'first_planner.json').open('x') as stream:
                json.dump(dict(iteration=player.state.game_loop // 4,
                    time_seconds=player.time, time_formatted=player.time_formatted,
                    bo_index=player.current_bo_index, bo_length=len(player.bo_data),
                    planner_ct=planner.ct, request=data), stream, ensure_ascii=False)
            print('PREFIX_BOUNDARY: first Planner request captured; no answer supplied', flush=True)
        return send(data)

    planner._send = capture


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "agent_system/environments/env_package/sc2_online_restored"))
    import envSC2

    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    target = os.getenv('SC2_RNG_PREFIX_TARGET', '120')
    assert target in ('120', 'first-planner')
    output = Path(os.environ['OUTPUT_DIR'])
    if target == 'first-planner':
        class BoundaryPlayer(envSC2.LLMPlayer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                record_first_planner(self, output)

        envSC2.LLMPlayer = BoundaryPlayer
    else:
        assert os.environ['SC2_GAME_TIME_LIMIT'] == '120'
    env = envSC2.SC2Env(config={})
    try:
        env.reset(seed=int(os.environ["SC2_ENV_SEED"]))
        if target == 'first-planner' and not env.game_end_flag:
            boundary = json.loads((output / 'first_planner.json').read_text())
            assert boundary['bo_index'] == boundary['bo_length'] == 237
            assert boundary['request']['obs'] == env.obs
            assert boundary['request']['last_consumed_policy_action_id'] is None
            (output / 'result.json').write_text(json.dumps(dict(
                status='reached_first_planner', planner_answers_sent=0,
                scope='intentional diagnostic stop, not a game outcome'), indent=2) + '\n')
        elif not env.game_end_flag:
            raise RuntimeError("Unexpected Planner request before prefix cap; no model answer sent")
        else:
            data = env.game_end_data
            assert data['last_consumed_policy_action_id'] is None, data
            env.env_a_process.join(timeout=20)
            assert env.env_a_process.exitcode == 0, env.env_a_process.exitcode
            (output / 'terminal.json').write_text(json.dumps(dict(
                scope='prefix diagnostic, NOT a win-rate evaluation',
                planner_answers_sent=0, raw_terminal=data), ensure_ascii=False, indent=2) + '\n')
            if target == 'first-planner':
                (output / 'result.json').write_text(json.dumps(dict(
                    status='ended_before_first_planner', planner_answers_sent=0)) + '\n')
    finally:
        env.close()
