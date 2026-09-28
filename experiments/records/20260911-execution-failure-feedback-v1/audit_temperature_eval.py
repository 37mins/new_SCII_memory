"""Read two finished temperature runs; print exact choices, queues and rewards.

Usage: python3 audit_temperature_eval.py RUNS_ROOT
No model loading, writes, game launch, or automatic promotion decision.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

SHA = '9ae8668c4c6045f5834d197c72f7495c175c1c73'
PREFIX = '20260911-execution-failure-feedback-v1-spb2-A4-l6-temp'


def audit(root):
    arms, configs = {}, {}
    for t in ('04', '08'):
        p = root / (PREFIX + t + '-eval8-s42613-r1')
        def read(name): return json.loads((p/name).read_text())
        def rows(name): return [json.loads(s) for s in (p/name).read_text().splitlines()]
        result, manifest, load = read('result.json'), read('manifest.json'), read('checkpoint_evaluation.json')
        assert result == read('.done') and result['status'] == 'completed'
        assert result['process_exit_code'] == result['training_updates'] == 0
        assert manifest['source_commit'] == SHA and not manifest['dirty']
        assert load['loaded_global_step'] == 4 and load['checkpoint_load_returned'] and load['val_only']
        assert load['game_seeds'] == rows('evaluation_seeds.jsonl')[0]['seeds'] == [42613]*8
        assert load['temperature'] == float('0.' + t[-1]) and load['feedback'] == '1'
        configs[t] = {k:v.replace(str(p),'<RUN>').replace(p.name,'<RUN>')
                      for k,v in read('effective_config.json')['overrides'].items()}
        endings = {r['env_index']:r for r in rows('episode_outcomes.jsonl')}
        assert set(endings) == set(range(8))
        mapping = {r['env']:r['game_pid'] for r in rows('game_worker_mapping.jsonl')}
        responses, games = rows('prompt_response.jsonl'), []
        for env, end in sorted(endings.items()):
            assert end['terminated'] and not end['truncated']
            active = [r for r in responses if r['env_idx']==env and r['step_index']<=end['step_index']]
            policy = [r for r in active if r['policy_action_applied']]
            assert len(policy) == end['policy_decision_count']
            logs = list((p/'sc2-logs'/f'game-{mapping[env]}').rglob('default_player_*.log'))
            assert len(logs)==1
            queues = []
            for line_no, line in enumerate(logs[0].read_text().splitlines(),1):
                match=re.search(r'\[INFO\]: \((\d+)\) refresh build order: (.*)',line)
                if match:
                    queues.append(dict(line=line_no, iteration=int(match[1]), actions=match[2].split(', ')))
            assert len(queues)==len(policy), (t,env,len(queues),len(policy))
            choices=[]
            for r,q in zip(policy,queues):
                choices.append(dict(step_index=r['step_index'],valid=r['is_action_valid'],
                    failure=r['failure_reason'],answer=r['llm_response'],
                    prompt_sha256=hashlib.sha256(r['prompt'].encode()).hexdigest(),
                    prompt_game_time=re.search(r'It is now (\d+:\d+)',r['prompt']).group(1),
                    queue=q,counts=dict(Counter(q['actions']))))
            games.append(dict(env=env,outcome=end['game_result'],reward=sum(r['reward'] for r in active),
                calls=len(policy),valid=sum(r['is_action_valid'] for r in policy),choices=choices,
                reward_events=[{k:r[k] for k in ('step_index','reward','done','policy_action_applied','is_action_valid')}
                               for r in active],
                executor_log=str(logs[0].relative_to(p))))
        average=sum(g['reward'] for g in games)/8
        assert abs(average-result['metrics']['val/text/test_score']) < .001
        early=[]
        for i in (0,1):
            choices=[g['choices'][i] for g in games if len(g['choices'])>i]
            valid=[c for c in choices if c['valid']]
            early.append(dict(call=i+1,count=len(choices),valid=len(valid),
                valid_input_hashes=dict(Counter(c['prompt_sha256'] for c in valid)),
                unique_valid_queues=len(set(tuple(c['queue']['actions']) for c in valid)),
                all_queue_patterns=dict(Counter(json.dumps(c['queue']['actions']) for c in choices))))
        calls=sum(g['calls'] for g in games); valid=sum(g['valid'] for g in games)
        arms[t]=dict(run=str(p),ended_at=result['ended_at'],wins=result['episode_outcomes']['wins'],
            mean_reward=average,calls=calls,valid=valid,early=early,games=games,
            missing_answer_close=sum('</answer>' not in c['answer'] for g in games for c in g['choices']),
            hashes={n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in
                    ['manifest.json','effective_config.json','result.json','.done','prompt_response.jsonl',
                     'episode_outcomes.jsonl','evaluation_seeds.jsonl','checkpoint_evaluation.json']})
    a,b=configs['04'],configs['08']
    diff={k:[a.get(k),b.get(k)] for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
    paths={k:diff.pop(k) for k in ('+ray_init._temp_dir','+ray_init.runtime_env.env_vars.TMPDIR',
                                  '+ray_init.runtime_env.env_vars.RAY_TMPDIR')}
    assert all(v==['/dev/shm/t04_0912_r1','/dev/shm/t08_0912_r1'] for v in paths.values())
    assert set(diff)=={'actor_rollout_ref.rollout.temperature','actor_rollout_ref.rollout.val_kwargs.temperature',
                       '+ray_init.runtime_env.env_vars.SC2_FEEDBACK_TEMPERATURE_AB'}
    return dict(source_sha=SHA,config_differences=diff,runtime_directory_differences=paths,arms=arms,
                limits='One independent game seed. Queue is dispatched intent, not proof of birth. '
                       'Missing answer tag is not an exact generation-budget clipping metric.')


if __name__=='__main__':
    print(json.dumps(audit(Path(sys.argv[1])),ensure_ascii=False,indent=2))
