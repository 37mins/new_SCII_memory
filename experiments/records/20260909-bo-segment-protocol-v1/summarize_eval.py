"""Read completed A/B artifacts; CPU tokenizer only, no model/game or writes."""
import hashlib
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer


def summarize(run):
    manifest = json.loads((run / "manifest.json").read_text())
    result = json.loads((run / "result.json").read_text())
    config = json.loads((run / "effective_config.json").read_text())["overrides"]
    response_budget = int(config["data.max_response_length"])
    rows = [json.loads(s) for s in (run / "prompt_response.jsonl").read_text().splitlines()]
    seeds = json.loads((run / "evaluation_seeds.jsonl").read_text())["seeds"]
    consumed = [r for r in rows if r["policy_action_applied"]]
    assert len({(r["env_idx"], r["policy_action_id"]) for r in consumed}) == len(consumed)
    tokenizer_path = Path(manifest["model_source"]) / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    lengths = [len(tokenizer.encode(r["llm_response"], add_special_tokens=False).ids)
               for r in consumed]
    games = []
    for env, seed in enumerate(seeds):
        episode = []
        for row in rows:
            if row["env_idx"] == env:
                episode.append(row)
                if row["done"]:
                    break
        assert episode and episode[-1]["done"]
        decisions = [r for r in episode if r["policy_action_applied"]]
        seconds = []
        for row in decisions:
            match = re.search(r"It is now (\d+):(\d+)", row["prompt"])
            assert match, "Actual query game time missing"
            seconds.append(int(match[1]) * 60 + int(match[2]))
        intervals = [b - a for a, b in zip(seconds, seconds[1:])]
        games.append(dict(env_index=env, seed=seed, won=bool(episode[-1]["won"]),
                          cumulative_reward=sum(r["reward"] for r in episode),
                          policy_decisions=len(decisions),
                          valid_decisions=sum(r["semantic_action_valid"] for r in decisions),
                          request_game_seconds=seconds,
                          median_interval_seconds=statistics.median(intervals) if intervals else None))
    artifacts = ["manifest.json", "effective_config.json", "result.json", ".done",
                 "episode_outcomes.jsonl", "prompt_response.jsonl", "evaluation_seeds.jsonl", "console.log"]
    return dict(run_id=run.name, run_path=str(run), source_sha=manifest["source_commit"],
                model_manifest_sha256=manifest["model_manifest_sha256"],
                gpu_selection=manifest["gpu_selection"], result_status=result["status"],
                started_at=manifest["started_at"], ended_at=result["ended_at"],
                process_exit_code=result["process_exit_code"], training_updates=result["training_updates"],
                wins=sum(g["won"] for g in games), games=games,
                no_policy_games=sum(g["policy_decisions"] == 0 for g in games),
                mean_cumulative_reward=statistics.mean(g["cumulative_reward"] for g in games),
                consumed_answers=len(consumed),
                format_valid=sum(r["is_action_valid"] for r in consumed),
                semantic_valid=sum(r["semantic_action_valid"] for r in consumed),
                failure_reasons=dict(Counter(r["failure_reason"] for r in consumed if r["failure_reason"])),
                decoded_text_token_length_histogram=dict(sorted(Counter(lengths).items())),
                response_budget=response_budget,
                decoded_text_at_budget=sum(n == response_budget for n in lengths),
                decoded_text_at_budget_invalid=sum(n == response_budget and not r["semantic_action_valid"]
                                                   for n, r in zip(lengths, consumed)),
                token_length_method="Retokenize consumed decoded text without special tokens; proxy only. Original generation token IDs/EOS flags were not saved here; not an exact truncation rate.",
                tokenizer_sha256=hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
                replay_paths=result["replays"],
                artifact_sha256={f: hashlib.sha256((run / f).read_bytes()).hexdigest() for f in artifacts})


if __name__ == "__main__":
    print(json.dumps([summarize(Path(p)) for p in sys.argv[1:]], indent=2, ensure_ascii=False))
