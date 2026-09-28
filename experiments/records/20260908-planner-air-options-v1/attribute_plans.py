"""Read a completed single-game run; attribute plans, not production or combat."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys


def attribute(rows, lines):
    plans = [(i + 1, s) for i, s in enumerate(lines) if 'BO ori:' in s and ', BO:' in s]
    consumed = [r for r in rows if r['policy_action_applied']]
    assert len(consumed) == len(plans), (len(consumed), len(plans))
    records = []
    counts = {'valid_model_plan': Counter(), 'invalid_rule_processed_plan': Counter()}
    for row, (line_number, line) in zip(consumed, plans):
        logged_raw, encoded = line.split('BO ori:', 1)[1].split(', BO:', 1)
        raw = row['llm_response']
        assert raw.replace('\n', ',').strip().rstrip(',') == logged_raw.strip().rstrip(',')
        final = ast.literal_eval(encoded.strip())
        assert isinstance(final, list) and all(isinstance(s, str) for s in final)
        valid = row['semantic_action_valid']
        assert row['semantic_action_evaluable']
        source = 'valid_model_plan' if valid else 'invalid_rule_processed_plan'
        builds = [s for s in final if not s.startswith('Launch OFFENSE')]
        raw_builds = [s.strip()[2:].strip() for s in raw.splitlines()
                      if s.strip().startswith('- ') and not s.strip().startswith('- Launch OFFENSE')]
        if valid:
            assert builds == raw_builds, (row['step_index'], builds, raw_builds)
        counts[source].update(builds)
        todo = row['prompt'].split('The remaining BO list in this phase:\n', 1)[1].split('Use only exact', 1)[0]
        offered = {m[1]: int(m[2]) for m in re.finditer(r'^- (.+?) x (\d+)$', todo, re.M)}
        added = Counter(builds) - Counter(raw_builds)
        records.append(dict(step_index=row['step_index'], source=source, console_line=line_number,
                            raw_answer=raw, final_plan=final,
                            air_offered={k: offered[k] for k in ('starport', 'vikingfighter') if k in offered},
                            air_added_not_literal_in_raw={k: added[k] for k in ('starport', 'vikingfighter') if added[k]}))
    return dict(consumed_answers=len(consumed), all_logged_rows=len(rows),
                valid_answers=sum(r['semantic_action_valid'] for r in consumed),
                cumulative_reward_all_rows=sum(r['reward'] for r in rows),
                plan_counts={k: dict(v) for k, v in counts.items()}, records=records,
                scope='Planner production plans only; Commander guard, actual commands, completions and combat require separate evidence. Invalid-row output is rule-processed, not model attribution.')


if __name__ == '__main__':
    folder = Path(sys.argv[1])
    result = json.loads((folder / 'result.json').read_text())
    names = ('console.log', 'prompt_response.jsonl', 'result.json', 'effective_config.json')
    hashes = {n: hashlib.sha256((folder / n).read_bytes()).hexdigest() for n in names}
    assert hashes['console.log'] == result['console_sha256']
    rows = [json.loads(s) for s in (folder / 'prompt_response.jsonl').read_text().splitlines()]
    report = attribute(rows, (folder / 'console.log').read_text().splitlines())
    print(json.dumps(dict(run=str(folder), hashes=hashes, report=report), ensure_ascii=False, indent=2))
