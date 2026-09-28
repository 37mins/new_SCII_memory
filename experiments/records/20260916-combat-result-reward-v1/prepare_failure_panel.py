"""Freeze three existing failed-game inputs; no prompt rewriting or model load."""
import hashlib
import json
from pathlib import Path
import sys

SEEDS = (1436145665, 363527354, 3434611143)


def prepare(root):
    names = ('manifest.json', 'result.json', '.done', 'evaluation_seeds.jsonl',
             'episode_outcomes.jsonl', 'prompt_response.jsonl', 'checkpoint_evaluation.json')
    raw = {name: (root/name).read_bytes() for name in names}
    result = json.loads(raw['result.json'])
    assert result == json.loads(raw['.done']) and result['status'] == 'completed'
    assert result['process_exit_code'] == 0
    seeds = json.loads(raw['evaluation_seeds.jsonl'])['seeds']
    rows = [json.loads(s) for s in raw['prompt_response.jsonl'].splitlines()]
    endings = {r['env_index']: r for r in map(json.loads, raw['episode_outcomes.jsonl'].splitlines())}
    cases = []
    for i, seed in enumerate(SEEDS):
        env = seeds.index(seed)
        end = endings[env]
        assert end['game_result'] == 'Defeat' and end['terminated'] and not end['truncated']
        policy = [(n, r) for n, r in enumerate(rows, 1) if r['env_idx'] == env
                  and r['step_index'] <= end['step_index'] and r['policy_action_applied']]
        assert len(policy) == 2
        line, row = policy[1]
        prompt = row['prompt']
        cases.append(dict(index=i, game_seed=seed, env=env, source_line=line,
                          prompt=prompt, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                          historical_answer=row['llm_response'],
                          sampling_seeds=list(range(12000+8*i, 12008+8*i))))
    manifest = json.loads(raw['manifest.json'])
    return dict(source=str(root), source_runtime_sha=manifest['source_commit'],
                artifact_sha256={k:hashlib.sha256(v).hexdigest() for k,v in raw.items()},
                cases=cases, games=0, optimizer_updates=0,
                note='Three diagnostic inputs selected because of known failures; not an independent success-rate test.')


if __name__ == '__main__':
    result = prepare(Path(sys.argv[1]))
    with Path(sys.argv[2]).open('x') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(json.dumps(dict(contexts=len(result['cases']), unchanged_input_hashes=[r['prompt_sha256'] for r in result['cases']])))
