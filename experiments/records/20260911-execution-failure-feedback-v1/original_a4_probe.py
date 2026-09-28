"""Reuse completed A4 answers; only generate the missing original-SpB arm."""
import argparse
import hashlib
import json
from pathlib import Path

from example_order_probe import load_cases, PROMPT_SHA


def prepare(source_run, paired_run):
    record = json.loads(Path(__file__).with_name("example-order-r1-results.json").read_text())
    for name, digest in record["artifact_sha256"].items():
        assert hashlib.sha256((paired_run/name).read_bytes()).hexdigest() == digest, name
    manifest = json.loads((paired_run/"manifest.json").read_text())
    assert manifest["source_commit"] == record["source_commit"]
    assert (paired_run/".done").read_text().startswith("EXIT_CODE=0\n")
    rows = [json.loads(line) for rank in range(4)
            for line in (paired_run/f"rank-{rank}.jsonl").read_text().splitlines()]
    starts = {r["seed"]:r for r in rows if r["event"] == "generation_start" and r["arm"] == "A"}
    answers = {r["seed"]:r for r in rows if r["event"] == "generation" and r["arm"] == "A"}
    assert set(starts) == set(answers) == set(range(12000, 12008))
    for row in rows:
        if row["event"] in ("model_ready", "arm_complete") and row["arm"] == "A":
            assert row["optimizer_counters"] == [15]
    assert all(r["prompt_sha256"] == PROMPT_SHA for r in answers.values())
    assert all(r["input_tokens"] == 1478 for r in starts.values())
    cases = load_cases(source_run/"prompt_response.jsonl")
    for case in cases:
        del case["prompts"]  # Do not expose the rejected, reordered B example.
    return cases, dict(run_id=record["run_id"], source_commit=manifest["source_commit"],
                       path=str(paired_run), artifact_sha256=record["artifact_sha256"],
                       starts=starts, answers=answers)


def compare(answers, reference):
    pairs = []
    for row in sorted(answers, key=lambda r:r["seed"]):
        old = reference["answers"][row["seed"]]
        def metrics(answer):
            closed = "</answer>" in answer["raw_answer"]
            return dict(valid=answer["valid"], depot_first=answer["depot_first"],
                        depot_before_supply_consumer=answer["depot_before_supply_consumer"],
                        eos=answer["eos_seen"], closed=closed,
                        complete_correction=bool(answer["depot_before_supply_consumer"] and answer["eos_seen"] and closed),
                        token_count=answer["token_count"], reason=answer["reason"])
        pairs.append(dict(seed=row["seed"], original=metrics(row), A4=metrics(old)))
    assert [p["seed"] for p in pairs] == list(range(12000,12008))
    return dict(pairs=pairs, reference_run=reference["run_id"],
                newly_generated_answers=8, reused_answers=8,
                comparison="same failure input and original example; only model weights differ")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--paired-a4-run", type=Path, required=True)
    args = parser.parse_args()
    cases, reference = prepare(args.source_run, args.paired_a4_run)
    print(json.dumps(dict(status="CPU reference check passed", cases=len(cases),
          unique_prompts=len({c["prompt"] for c in cases}), paired_run=reference["run_id"],
          prompt_sha256=PROMPT_SHA, seeds=sorted(reference["answers"]),
          new_GPU_generations=0, games=0, optimizer_updates=0), indent=2))


if __name__ == "__main__":
    main()
