"""CPU audit of retained seed42613 replays; not an exact PPO re-score.

Uses replay PlayerStats (sampled counters, not assumed to be live observations).
No inferred unit prices, command attribution, model load or new SC2 game.
"""
import hashlib
import json
from pathlib import Path
import runpy
import statistics
import sys

from mpyq import MPQArchive
from s2protocol.versions import build, latest

CATEGORIES = ('Army', 'Economy', 'Technology')
SCALE = .001


def values(stats):
    return {kind.lower(): sum(stats[f'm_scoreValue{resource}{kind}{category}']
                             for resource in ('Minerals', 'Vespene') for category in CATEGORIES)
            for kind in ('Killed', 'Lost')}


def main(snapshot):
    old = Path(__file__).parent.parent / '20260911-execution-failure-feedback-v1'
    report = runpy.run_path(str(old / 'audit_temperature_eval.py'))['audit'](snapshot)
    timeline = json.loads((old / 'temperature-timeline.json').read_text())
    games = []
    for arm, arm_result in report['arms'].items():
        root = Path(arm_result['run'])
        for game in arm_result['games']:
            replay = root / Path(game['executor_log']).parent / 'replay.SC2Replay'
            archive = MPQArchive(str(replay))
            header = latest().decode_replay_header(archive.header['user_data_header']['content'])
            assert header['m_version']['m_baseBuild'] == 75689
            proto = build(75689)
            stats = [e for e in proto.decode_replay_tracker_events(archive.read_file('replay.tracker.events'))
                     if e['_event'].endswith('SPlayerStatsEvent') and e['m_playerId'] == 1]
            sha = hashlib.sha256(replay.read_bytes()).hexdigest()
            expected = json.loads((root / 'result.json').read_text())['replays'][str(replay.relative_to(root))]
            assert sha == expected
            counters = [dict(loop=e['_gameloop'], **values(e['m_stats'])) for e in stats]
            for before, after in zip(counters, counters[1:]):
                assert after['killed'] >= before['killed'] and after['lost'] >= before['lost']
            first, final = counters[0], counters[-1]
            net = (final['killed'] - first['killed']) - (final['lost'] - first['lost'])
            delta_sum = sum((b['killed']-a['killed'])-(b['lost']-a['lost'])
                            for a, b in zip(counters, counters[1:]))
            assert net == delta_sum
            first_plan_loop = game['choices'][0]['queue']['iteration'] * 4
            # The first replay score is at loop1; all opening queries are before
            # the first kill/loss, checked explicitly rather than interpolated.
            before_plan = [r for r in counters if r['loop'] <= first_plan_loop]
            baseline = before_plan[-1] if before_plan else first
            assert baseline['killed'] == baseline['lost'] == 0
            games.append(dict(arm=arm, env=game['env'], replay=str(replay), replay_sha256=sha,
                              second_plan=game['choices'][1]['counts'], calls=game['calls'],
                              old_return=game['reward'], outcome=game['outcome'],
                              first_stats=first, last_stats=final, net_value=net,
                              candidate_component=SCALE*net, counter_sum_check=True))
    summary = {}
    for arm in report['arms']:
        selected = [g for g in games if g['arm'] == arm]
        summary[arm] = dict(old_return_unique=len({g['old_return'] for g in selected}),
                            combat_unique=len({g['net_value'] for g in selected}),
                            net_min=min(g['net_value'] for g in selected),
                            net_max=max(g['net_value'] for g in selected),
                            candidate_std=statistics.pstdev(g['candidate_component'] for g in selected))
    return dict(scope='Historical A4/feedback-on temperature games, not new feedback-off RL.',
                scale=SCALE, categories=list(CATEGORIES), games=games, summary=summary,
                source_audit=report['source_sha'],
                timeline_sha256=hashlib.sha256((old/'temperature-timeline.json').read_bytes()).hexdigest(),
                limits=['Replay stats are sampled; this is not exact live-boundary reward or GiGPO advantage.',
                        'All games lost; differing combat score does not establish which policy improves winning.',
                        'Identical counters give identical reward; this cannot create exploration.',
                        'Historical temperature/feedback differs from the proposed RL; no transfer guarantee.'])


if __name__ == '__main__':
    result = main(Path(sys.argv[1]))
    with Path(sys.argv[2]).open('x') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(json.dumps(result['summary'], indent=2))
