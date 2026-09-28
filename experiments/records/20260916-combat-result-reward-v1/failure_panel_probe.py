"""CPU input checks and empirical plan frequencies for the approved 48 answers."""
from collections import Counter
import hashlib
import json
from pathlib import Path


def load_panel(path):
    panel = json.loads(Path(path).read_text())
    assert len(panel['cases']) == 3
    cases = []
    for index, case in enumerate(panel['cases']):
        assert case['index'] == index
        assert hashlib.sha256(case['prompt'].encode()).hexdigest() == case['prompt_sha256']
        assert case['sampling_seeds'] == list(range(12000+8*index, 12008+8*index))
        for draw in range(2):
            cases.append(dict(case, draw_group=draw,
                              sampling_seeds=case['sampling_seeds'][4*draw:4*draw+4]))
    return panel, cases


def summarize_panel(rows, panel):
    answers = [r for r in rows if r['event'] == 'generation']
    starts = [r for r in rows if r['event'] == 'generation_start']
    expected = {(arm, case['index'], seed) for arm in ('original', 'A4')
                for case in panel['cases'] for seed in case['sampling_seeds']}
    key = lambda r: (r['arm'], r['case'], r['seed'])
    assert len(answers) == len(starts) == len(expected) == 48
    assert set(map(key, answers)) == set(map(key, starts)) == expected
    starts = {key(r): r for r in starts}
    report = []
    for case in panel['cases']:
        i = case['index']
        for seed in case['sampling_seeds']:
            a, b = starts['original', i, seed], starts['A4', i, seed]
            assert a['input_token_sha256'] == b['input_token_sha256']
            assert a['cuda_rng_sha256'] == b['cuda_rng_sha256']
        arms = {}
        for arm in ('original', 'A4'):
            selected = [r for r in answers if r['arm'] == arm and r['case'] == i]
            assert all(r['prompt_sha256'] == case['prompt_sha256'] for r in selected)
            valid = [r for r in selected if r['valid']]
            plans = Counter(tuple(r['parsed_queue']) for r in valid)
            arms[arm] = dict(answers=8, valid=len(valid), legal_plan_kinds=len(plans),
                marine42=sum(r['parsed_queue'].count('marine') == 42 for r in valid),
                no_eos=sum(not r['eos_seen'] for r in selected),
                no_closing_tag=sum('</answer>' not in r['raw_answer'] for r in selected),
                sampling_distribution=[dict(queue=list(queue), count=n, fraction_of_all_answers=n/8)
                                       for queue, n in plans.most_common()])
        report.append(dict(case=i, game_seed=case['game_seed'], arms=arms))
    return dict(contexts=report, paired_input_and_sampling_rng=True,
                interpretation='Empirical frequencies from eight draws per condition, not exact vocabulary probabilities or game success.')
