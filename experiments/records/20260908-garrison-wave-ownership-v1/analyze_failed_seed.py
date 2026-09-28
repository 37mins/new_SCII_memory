"""Offline audit of downloaded seed43567 logs. No model, game, or server writes.

Usage: python analyze_failed_seed.py SNAPSHOT_ROOT OUTPUT_JSON
The original/ folder is the first manifest/config snapshot; original-final/
contains the completed evaluation logs. after-step1/ contains the failed RL run.
Worker .out files are copied separately into each final arm's workers/ folder.
"""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys


def read_jsonl(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def observation(prompt):
    match = re.search(r'It is now ([0-9:]+).*?with (\d+) minerals and (\d+) vespene.*?(\d+) is allocated to the army, (\d+) to workers', prompt)
    time, minerals, gas, army, workers = match.groups()
    own = prompt.split('In terms of our military forces, we have ')[1].split('. In terms of the structures')[0]
    structures = prompt.split('In terms of the structures, we have ')[1].split('. \nIn terms of research')[0]
    enemy = prompt.split('As for the visible enemy presence, ')[1].split('As for the visible enemy structures')[0]
    todo = prompt.split('The remaining BO list in this phase:\n')[1].split('Use only exact')[0].strip()
    return dict(time=time, minerals=int(minerals), gas=int(gas), army_supply=int(army),
                worker_supply=int(workers), own=own, structures=structures, enemy=enemy,
                todo=todo, no_research='we have researched no research currently' in prompt)


def normalized_answer(row):
    return row['llm_response'].split('</think>')[-1].replace('\n', ',')


def worker_evidence(folder, reset_index, episodes):
    result = {}
    for path in sorted((folder / 'workers').glob('*.out')):
        text = re.sub(r'\x1b\[[0-9;]*m', '', path.read_text())
        resets = list(re.finditer(r'Acquired SC2 reset slot (\d+)/8\.', text))
        if not resets:
            continue
        start = resets[reset_index].start()
        end = resets[reset_index+1].start() if reset_index+1 < len(resets) else len(text)
        offset = text[:start].count('\n')
        lines = text[start:end].splitlines()
        plans = [(n, m) for n, line in enumerate(lines, offset+1)
                 if (m := re.fullmatch(r'BO ori: (.*), BO: (\[.*\])', line))]
        matched = [i for i, episode in episodes.items()
                   if [normalized_answer(r) for r in episode['decisions']] == [m[1] for _, m in plans]]
        assert len(matched) == 1, (path, 'ambiguous response-to-worker mapping', matched, len(plans))
        env = matched[0]
        # Reset slots are concurrency-gate slots, not stable environment IDs.
        # Match the complete ordered answer sequence to the explicit env ledger.
        first = next(n for n, line in enumerate(lines, offset+1) if line == 'Generate new plan... [0]')
        fixed_starts, fixed_commands, samples, events, commands, skips = [], [], {}, [], [], []
        current_frame = None
        for n, line in enumerate(lines, offset+1):
            frame_match = re.search(r'\[INFO\]: \((\d+)\) ([^:]+): (.*)', line)
            if frame_match:
                current_frame, key, value = int(frame_match[1]), frame_match[2], frame_match[3]
                if key in {'time_seconds', 'minerals', 'vespene', 'supply_army', 'supply_workers'}:
                    samples.setdefault(current_frame, {})[key] = float(value)
                if key.startswith('llm_decision_final_action_'):
                    actions = ast.literal_eval(value)
                    item = dict(line=n, frame=current_frame, seconds=round(current_frame*4/22.4, 2), actions=actions)
                    commands.append(item)
                    if n < first:
                        fixed_commands.append([current_frame, actions])
                if key.startswith('bo_skipped_'):
                    skips.append(dict(line=n, frame=current_frame, value=ast.literal_eval(value)))
            if n < first and (m := re.search(r'\[(\d+:\d+)\] >>> Start Processing BO #(\d+): (.*?) <<<', line)):
                fixed_starts.append([m[1], int(m[2]), m[3]])
            if any(k in line for k in ('[Auto-Attack]', '已创建总攻波次', '集结完毕', '已全灭', '取消本波', 'Opening marine guard')):
                events.append(dict(line=n, frame=current_frame, text=line))
        assert [r[1] for r in fixed_starts] == list(range(237)), (path, len(fixed_starts))
        queues = []
        for j, ((line_no, match), row) in enumerate(zip(plans, episodes[env]['decisions'])):
            stop = plans[j+1][0] if j+1 < len(plans) else offset+len(lines)+1
            section = lines[line_no-offset-1:stop-offset-1]
            refreshed = next((s.split('refresh build order: ', 1)[1] for s in section if 'refresh build order: ' in s), None)
            assert refreshed is not None, (path, j)
            queues.append(dict(call=j+1, game_time=observation(row['prompt'])['time'], line=line_no,
                valid=row['is_action_valid'], raw=row['llm_response'], planner_bo=ast.literal_eval(match[2]),
                executor_queue=refreshed, command_events=[c for c in commands if line_no < c['line'] < stop],
                skips=[s for s in skips if line_no < s['line'] < stop]))
        terminal_time = re.findall(r'time_cost: (\d+)', '\n'.join(lines))
        result[env] = dict(worker=str(path), worker_sha256=sha(path.read_bytes()),
            game_seconds=int(terminal_time[-1]) if terminal_time else None,
            fixed_bo_starts=fixed_starts, fixed_commands=fixed_commands,
            preplanner_auto_attacks=[e for e in events if e['line']<first and '[Auto-Attack]' in e['text']],
            postplanner_auto_attacks=[e for e in events if e['line']>first and '[Auto-Attack]' in e['text']],
            wave_events=events, stock_snapshots=[dict(frame=f, **s) for f,s in samples.items() if len(s)==5],
            queues=queues, skips=skips)
    assert len(result)==8
    return result


def main(root):
    reports = {}
    first_prompts = []
    for name, reset in [('original-final',0), ('after-step1',1)]:
        folder=root/name
        metadata_folder=root/('original' if reset==0 else name)
        manifest=json.loads((metadata_folder/'manifest.json').read_text())
        assert manifest['source_commit']=='e1e25ce1d14362b209e34ec73871ebaeb63538a6' and manifest['formal'] and not manifest['dirty']
        rows=read_jsonl(folder/'prompt_response.jsonl')
        outcomes=read_jsonl(folder/'episode_outcomes.jsonl')
        lo=0 if reset==0 else max(o['step_index'] for o in outcomes if o['reset_index']==reset-1)+1
        selected_outcomes=[o for o in outcomes if o['reset_index']==reset]
        hi=max(o['step_index'] for o in selected_outcomes)
        rows=[r for r in rows if lo<=r['step_index']<=hi]
        episodes={}
        for i in range(8):
            outcome=next(o for o in selected_outcomes if o['env_index']==i)
            active=[r for r in rows if r['env_idx']==i and r['step_index']<=outcome['step_index']]
            decisions=[r for r in active if r['policy_action_applied']]
            assert active[-1]['done'] and len(decisions)==outcome['policy_decision_count']
            assert len(set(r['policy_action_id'] for r in decisions))==len(decisions)
            first_prompts.append(decisions[0]['prompt'])
            valid=sum(r['is_action_valid'] for r in decisions)
            episodes[i]=dict(outcome=outcome, decisions=decisions, decision_count=len(decisions),
                valid_count=valid, cumulative_logged_reward=sum(r['reward'] for r in active),
                first=observation(decisions[0]['prompt']), last=observation(decisions[-1]['prompt']),
                first_enemy_air=next((observation(r['prompt'])['time'] for r in decisions
                    if any(t in observation(r['prompt'])['enemy'] for t in ('viking','battlecruiser','banshee','liberator'))),None))
        evidence=worker_evidence(folder,reset,episodes)
        decisions=[r for e in episodes.values() for r in e['decisions']]
        todos=[observation(r['prompt'])['todo'].lower() for r in decisions]
        for i,e in episodes.items():
            e['worker_evidence']=evidence[i]
            del e['decisions']
        reports[name]=dict(run_manifest=manifest, step_index_interval=[lo,hi], episodes=episodes,
            wins=sum(o['game_result']=='Victory' for o in selected_outcomes), games=len(selected_outcomes),
            decisions=len(decisions), valid=sum(r['is_action_valid'] for r in decisions),
            errors=dict(Counter(r['failure_reason'] for r in decisions if not r['is_action_valid'])),
            all_six_marine_answers=sum([s.strip() for s in r['llm_response'].splitlines() if s.strip().startswith('-')]==['- marine']*6 for r in decisions),
            upgrade_or_support_todo_occurrences={name:sum(name in t for t in todos)
                for name in ('stimpack','combatshield','terraninfantryweapons','medivac')},
            all_no_research=all(observation(r['prompt'])['no_research'] for r in decisions),
            mean_cumulative_logged_reward=sum(e['cumulative_logged_reward'] for e in episodes.values())/8,
            input_hashes={p.name:sha(p.read_bytes()) for p in folder.iterdir() if p.is_file()})
    a,b=reports.values()
    assert len(set(first_prompts))==1
    for key in ('source_commit','source_files_sha256','model_manifest_sha256','bo_sha256'):
        assert a['run_manifest'][key]==b['run_manifest'][key],key
    fixed=[e['worker_evidence']['fixed_commands'] for r in reports.values() for e in r['episodes'].values()]
    starts=[e['worker_evidence']['fixed_bo_starts'] for r in reports.values() for e in r['episodes'].values()]
    return dict(scope='Read-only historical comparison; same seed, not matched model sampling RNG/hardware; no trained checkpoint available.',
        runtime_sha=a['run_manifest']['source_commit'], game_seed=43567,
        first_prompt_sha256=sha(first_prompts[0].encode()), all_16_first_prompts_identical=True,
        all_16_fixed_start_times_and_actions_identical=all(x==starts[0] for x in starts),
        all_16_logged_fixed_commands_identical=all(x==fixed[0] for x in fixed), reports=reports)


if __name__=='__main__':
    result=main(Path(sys.argv[1]))
    Path(sys.argv[2]).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    for name,r in result['reports'].items():
        print(name,{k:r[k] for k in ('wins','games','decisions','valid','mean_cumulative_logged_reward','all_six_marine_answers','all_no_research')})
        print('per_env',[(i,e['decision_count'],e['valid_count'],round(e['cumulative_logged_reward'],3),e['worker_evidence']['game_seconds']) for i,e in r['episodes'].items()])
    print({k:v for k,v in result.items() if k!='reports'})
