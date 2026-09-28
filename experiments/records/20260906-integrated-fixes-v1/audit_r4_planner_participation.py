"""Read r4 paired-evaluation logs only; no model imports, GPU, or server writes."""
import ast
import hashlib
import json
from pathlib import Path
import re

ROOT = Path('/mnt/data0601/yangjinyue/new_SCII/experiments/20260906-integrated-fixes-v1/runs')
SUITE = '20260906-integrated-fixes-v1-qwen-instruct2507-l6-8gpu-fp32-rl96-gate8-s0-r4'
SOURCE = Path('/root/new-scii-empty-bo-frame-20260907/code-ca7885d')
SHA = 'ca7885d886aaaafb39fc105b1d5d46cddf4bcc5a'
PIDS = {'initial-eval': '3652008', 'step8-eval': '3954027'}


def read_rows(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prompt_time(prompt):
    match = re.search(r'It is now (\d+:\d+) in the game', prompt)
    return match[1] if match else None


def numbered(lines, start, stop):
    return [{'line': i + 1, 'text': lines[i]} for i in range(start, stop)]


reports = {}
fixed_starts = {}
for stage, pid in PIDS.items():
    run = ROOT / f'{SUITE}-{stage}'
    manifest = json.loads((run / 'manifest.json').read_text())
    assert manifest['source_commit'] == SHA and not manifest['dirty']
    assert manifest['mode'] == 'evaluation'
    for relative, expected in manifest['source_files_sha256'].items():
        assert digest(SOURCE / relative) == expected, relative
    bo_path = Path(manifest['bo_path'])
    assert digest(bo_path) == manifest['bo_sha256']
    fixed_bo = json.loads(bo_path.read_text())
    assert len(fixed_bo) == 237
    done = json.loads((run / '.done').read_text())
    assert done['status'] == 'completed' and done['process_exit_code'] == 0
    rows = read_rows(run / 'prompt_response.jsonl')
    seeds = read_rows(run / 'evaluation_seeds.jsonl')[0]['seeds']
    assert seeds[3] == 2621640384
    outcomes = read_rows(run / 'episode_outcomes.jsonl')
    episodes = []
    for outcome in sorted(outcomes, key=lambda x: x['env_index']):
        index = outcome['env_index']
        active = [x for x in rows if x['env_idx'] == index and x['step_index'] <= outcome['step_index']]
        applied = [x for x in active if x['policy_action_applied']]
        assert len(applied) == outcome['policy_decision_count']
        assert active[-1]['done']
        prompt = applied[0]['prompt'] if applied else None
        episodes.append(dict(env=index, seed=seeds[index], calls=len(applied),
            first_planner_game_time=prompt_time(prompt) if prompt else None,
            first_planner_units=prompt.split('In terms of our military forces, we have ')[-1].split('. In terms')[0] if prompt else None,
            nonterminal_reward=sum(x['reward'] for x in active if not x['done']),
            terminal_reward=active[-1]['reward'], cumulative_reward=sum(x['reward'] for x in active)))
    worker_paths = list((Path(f'/dev/shm/sc2_real_r4_{stage}') / 'session_latest/logs').glob(f'worker-*-{pid}.out'))
    assert len(worker_paths) == 1
    worker = worker_paths[0].resolve()
    lines = [re.sub(r'\x1b\[[0-9;]*m', '', x) for x in worker.read_text().splitlines()]
    plan_indices = [i for i, x in enumerate(lines) if re.fullmatch(r'Generate new plan\.\.\. \[\d+\]', x)]
    outcome = next(x for x in outcomes if x['env_index'] == 3)
    applied = [x for x in rows if x['env_idx'] == 3 and x['policy_action_applied'] and x['step_index'] <= outcome['step_index']]
    assert len(plan_indices) == len(applied)
    starts = []
    for i, line in enumerate(lines):
        match = re.search(r'\[(\d+:\d+)\] >>> Start Processing BO #(\d+): (.*?) <<<', line)
        if match:
            starts.append(dict(line=i + 1, time=match[1], index=int(match[2]), bo=match[3]))
    fixed = [x for x in starts if x['line'] <= plan_indices[0]]
    assert [x['index'] for x in fixed] == list(range(237))
    assert [x['bo'] for x in fixed] == [x['action'] for x in sorted(fixed_bo, key=lambda x: tuple(map(int, x['time'].split(':'))))]
    fixed_starts[stage] = fixed
    timeline = []
    for number, (start, response) in enumerate(zip(plan_indices, applied)):
        stop = plan_indices[number + 1] if number + 1 < len(plan_indices) else len(lines)
        parsed = [(i, m) for i in range(start, stop) if (m := re.fullmatch(r'BO ori: (.*), BO: (\[.*\])', lines[i]))]
        assert len(parsed) == 1
        parsed_line, match = parsed[0]
        assert match[1] == response['llm_response'].split('</think>')[-1].replace('\n', ','), (stage, number)
        queue = ast.literal_eval(match[2])
        actual = [x for x in starts if start < x['line'] <= stop]
        assert actual[0]['time'] == prompt_time(response['prompt'])
        timeline.append(dict(call=number + 1, game_time=actual[0]['time'], request_line=start + 1,
            raw_answer=response['llm_response'], queue=queue, queue_line=parsed_line + 1,
            started_actions=actual, finished_or_skipped=sum('Completed/Skipped.' in x for x in lines[start:stop]),
            timeout_count=sum('[Skip] BO #' in x for x in lines[start:stop]),
            reward=response['reward'], done=response['done']))
    evidence_keys = ('Generate new plan', 'refresh build order:', 'time_cost:', 'game_result:',
        '[Wait]', '[Skip]', 'bo_skipped_', 'BO ori:', 'Start Processing BO', 'Completed/Skipped.')
    post_events = [dict(line=i + 1, text=x) for i, x in enumerate(lines)
        if i >= plan_indices[0] and any(key in x for key in evidence_keys)]
    result_seconds = int(next(re.search(r'time_cost: (\d+)', x)[1] for x in reversed(lines) if 'time_cost:' in x))
    reports[stage] = dict(run=str(run), worker=str(worker), worker_sha256=digest(worker),
        frozen_source=SHA, bo_path=str(bo_path), bo_sha256=digest(bo_path), fixed_bo_entries=237,
        source_hashes=manifest['source_files_sha256'], episodes=episodes,
        target_game_time_seconds=result_seconds, fixed_end_evidence=numbered(lines, plan_indices[0] - 10, plan_indices[0] + 2),
        fixed_first_scout=next(dict(line=i + 1, text=x) for i, x in enumerate(lines[:plan_indices[0]]) if '正在派遣 SCV' in x),
        auto_attacks_before_planner=[dict(line=i + 1, text=x) for i, x in enumerate(lines[:plan_indices[0]]) if '[Auto-Attack]' in x],
        timeline=timeline, post_events=post_events if stage == 'step8-eval' else post_events[-14:],
        first_prompt=applied[0]['prompt'], last_prompt=applied[-1]['prompt'],
        artifact_hashes={name: digest(run / name) for name in ('manifest.json', 'effective_config.json', '.done', 'evaluation_seeds.jsonl', 'prompt_response.jsonl', 'episode_outcomes.jsonl')})

a, b = (fixed_starts[x] for x in PIDS)
differences = [(x, y) for x, y in zip(a, b) if (x['time'], x['bo']) != (y['time'], y['bo'])]
assert differences[0][0]['index'] == differences[0][1]['index'] == 10
assert [sum(x['calls'] for x in reports[k]['episodes']) for k in PIDS] == [306, 96]
print(json.dumps(dict(scope='Read-only CPU log audit; no game, model or GPU', target_seed=2621640384,
    first_fixed_bo_timing_divergence=differences[0], reports=reports), ensure_ascii=False, indent=2))
