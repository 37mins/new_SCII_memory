"""Read existing r3/B4 evidence only; no model, game, reward or runtime changes."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
RECORD = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
from test_r6_format_reward import strict_validator

MARKER = 'Execution feedback for the previous segment (not owned units/buildings):\n'


def read_rows(path):
    return [dict(json.loads(line), source_line=i) for i, line in
            enumerate(path.read_text().splitlines(), 1)]


def feedback(prompt):
    return json.JSONDecoder().raw_decode(prompt.split(MARKER, 1)[1].lstrip())[0]


def first_bullet(answer):
    return next((line.strip() for line in answer.splitlines() if line.strip().startswith('- ')), None)


def end_evidence(starts, completed, final_structures, terminal):
    """Distinguish unfinished at observation end from a command rejected/not issued."""
    if completed:
        return 'completed_link_present'
    if starts:
        observed = {u['tag']: u for u in final_structures}
        if terminal and all(s['tag'] in observed and 0 < observed[s['tag']]['progress'] < 1
                            for s in starts):
            return 'unfinished_at_natural_game_end'
        return 'start_only_completion_not_established'
    return 'no_attributed_depot_start'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(snapshot, association_audit, b4_snapshot):
    registered = json.loads((RECORD / 'commandtrace-r3-results.json').read_text())
    assert digest(association_audit) == registered['artifact_sha256']['independent_association_audit.json']
    audit = json.loads(association_audit.read_text())
    for name, sha in audit['input_sha256'].items():
        assert digest(snapshot / name) == sha, name
    manifest = json.loads((snapshot / 'manifest.json').read_text())
    assert manifest['source_commit'] == registered['runtime_sha'] and not manifest['dirty']
    # The CPU validator is the actual registered runtime method, not a new grading rule.
    manager = 'agent_system/environments/strategic_sc2_env_manager_v2.py'
    assert digest(ROOT / manager) == manifest['source_files_sha256'][manager]
    fixture = json.loads((RECORD / 'supply-order-inputs.json').read_text())
    frozen = subprocess.check_output(['git', 'show', f"{registered['runtime_sha']}:experiments/records/20260911-execution-failure-feedback-v1/supply-order-inputs.json"], cwd=ROOT)
    assert json.loads(frozen) == fixture
    prompts = read_rows(snapshot / 'prompt_response.jsonl')
    mapping = {r['game_pid']: r for r in read_rows(snapshot / 'game_worker_mapping.jsonl')}
    interventions = read_rows(snapshot / 'interventions.jsonl')
    discarded = []
    with patch.dict(os.environ, SC2_ORIGINAL_BO_SEGMENT='1', SC2_PLAN_BULLET_MAX='6', SC2_PLAN_SEGMENT_MAX='6'):
        validate = strict_validator()
        for r in interventions:
            if r['call_index'] != 2:
                continue
            prompt = fixture['rows'][2]['prompt']
            assert hashlib.sha256(prompt.encode()).hexdigest() == r['prompt_sha256']
            valid, reason = validate(prompt, r['raw_generated'])
            discarded.append(dict(env=r['env'], source_line=r['source_line'],
                answer=r['raw_generated'], forwarded=r['forwarded'], valid=valid, reason=reason,
                first_bullet=first_bullet(r['raw_generated']), executed=False))
    online = []
    for f in sorted(snapshot.glob('game-*-command-trace.jsonl')):
        rows = read_rows(f)
        env = mapping[rows[0]['game_pid']]['env']
        received = {r['policy_action_id']: r for r in rows if r['event'] == 'received_answer'}
        boundaries = read_rows(snapshot / f"game-{rows[0]['game_pid']}-boundaries.jsonl")
        for action_id, answer in received.items():
            if answer['transport_origin'] != 'online_model_transport':
                continue
            prompt_row = next(r for r in prompts if r['env_idx'] == env and
                              r['policy_action_applied'] and r['policy_action_id'] == action_id)
            assert prompt_row['llm_response'] == answer['answer']
            fail = feedback(prompt_row['prompt'])
            failed_answer = received[fail['source_policy_action_id']]
            commands = [r for r in rows if r['event'] == 'local_command' and
                        r['origin']['policy_action_id'] == action_id]
            command_ids = {r['event_id'] for r in commands}
            starts = [r for r in rows if r['event'] == 'depot_started' and r['command_id'] in command_ids]
            completions = [r for r in rows if r['event'] == 'depot_completed' and
                           r['command_id'] in command_ids and not r['duplicate']]
            queue = next(r for r in rows if r['event'] == 'executor_queue' and
                         r['latest_policy_action_id'] == action_id)
            final = boundaries[-1]
            terminal_rows = [r for r in prompts if r['env_idx'] == env]
            total = 0
            for r in terminal_rows:
                total += r['reward']
                if r['done']:
                    break
            online.append(dict(env=env, game_pid=rows[0]['game_pid'], trace_file=f.name,
                source_line=prompt_row['source_line'], received_event=answer['event_id'],
                query_sha256=answer['prompt_sha256'], received_seconds=answer['time_seconds'],
                failure=fail, failure_origin=failed_answer['transport_origin'],
                answer=answer['answer'], valid=answer['valid'],
                executor_queue=[o['action'] for o in queue['origins']],
                command_events=commands, start_events=starts, completion_events=completions,
                final_time=final['time'], final_bo_index=final['bo_index'],
                final_depots=[u for u in final['structures'] if u['name']=='SUPPLYDEPOT'],
                new_marine_births=sum(r['event']=='marine_created' and r['event_id']>answer['event_id'] for r in rows),
                boundary_verdict=end_evidence(starts, completions, final['structures'], final['done']),
                cumulative_reward=total, response_reward=prompt_row['reward']))
    b4 = json.loads((RECORD / 'b4-training-signal.json').read_text())
    for name, sha in b4['artifact_hashes'].items():
        assert digest(b4_snapshot / name) == sha, name
    b4_raw = {r['source_line']: r for r in read_rows(b4_snapshot / 'prompt_response.jsonl')}
    b4_supply = []
    for event in b4['feedback_events']:
        if 'Supply blocked' not in event['failed']['reason']:
            continue
        r = b4_raw[event['source_line']]
        assert r['policy_action_applied'] and r['llm_response'] == event['answer']
        assert feedback(r['prompt'])['failed'] == event['failed']
        b4_supply.append(dict(source_line=event['source_line'], global_step=event['global_step'],
            env=event['env'], answer=event['answer'], reward=event['episode_reward'],
            depot_first=event['depot_is_first_bullet'],
            completion_evidence='historical_prompt_logs_do_not_establish_command_owned_completion'))
    summary = dict(runtime_sha=registered['runtime_sha'], games=8, independent_scenarios=1,
        discarded_third_answers=len(discarded), discarded_third_valid=sum(r['valid'] for r in discarded),
        discarded_third_depot_first=sum(r['first_bullet'].startswith('- supplydepot') for r in discarded),
        online_answers=len(online), online_depot_first=sum(r['executor_queue'][0]=='supplydepot' for r in online),
        online_depot_starts=sum(len(r['start_events']) for r in online),
        online_depot_completions=sum(len(r['completion_events']) for r in online),
        online_boundary_verdicts=dict(Counter(r['boundary_verdict'] for r in online)),
        online_failure_origins=dict(Counter(r['failure_origin'] for r in online)),
        online_reward_values=sorted({r['cumulative_reward'] for r in online}),
        b4_supply_feedback=len(b4_supply), b4_depot_first=sum(r['depot_first'] for r in b4_supply),
        new_gpu_runs=0, reward_enabled=False,
        conclusion='No new implementation bug established; proposed recovery bonus remains unaccepted.')
    return dict(summary=summary, r3_input_sha256=audit['input_sha256'],
                b4_input_sha256=b4['artifact_hashes'], discarded_third_answers=discarded,
                online_answers=sorted(online,key=lambda r:r['env']), b4_supply_feedback=b4_supply)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--association-audit', type=Path, required=True)
    parser.add_argument('--b4-snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.snapshot, args.association_audit, args.b4_snapshot)
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(result['summary'], ensure_ascii=False, indent=2))
