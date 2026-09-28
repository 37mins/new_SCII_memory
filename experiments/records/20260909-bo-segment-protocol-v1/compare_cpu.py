"""Print raw answers and A/B queue evidence. No model, game or GPU.

Usage: python compare_cpu.py SC2_online.zip prompt_response.jsonl
"""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
from audit_cpu import input_context, load_original
from test_original_bo_segment import BASELINE, pipeline


def main():
    archive, log = map(Path, sys.argv[1:])
    original, original_sha = load_original(archive)
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    raw, later = rows[37], rows[45]
    fixtures = [
        ("actual_29_marine_remainder8", raw["llm_response"], raw, 1, "unaltered log line38"),
        ("actual_six_actions", later["llm_response"], later, 2, "unaltered log line46"),
        ("derived_seven_marine", "\n".join(["- marine"] * 7), later, 2, "first7 actions of line38; line46 context, remainder42"),
        ("derived_fifteen_marine", "\n".join(["- marine"] * 15), later, 2, "first15 actions of line38; line46 context, remainder42"),
        ("derived_count29", "- marine x 29", later, 2, "lossless count encoding of line38; line46 context, remainder42"),
        ("derived_mixed_18_actions", "- supplydepot x 2\n- barracks\n- marine x 15", later, 2, "synthetic count boundary, real line46 candidates"),
        ("derived_bad_name", "- barrack\n- marine x 2", later, 2, "synthetic exact-name negative control"),
    ]
    result = dict(status="CPU only; GPU evaluation not started", baseline_sha=BASELINE,
                  archive_planner_sha256=original_sha,
                  log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(), cases=[])
    queries = {}

    def compact(trace):
        query = trace.pop("query")
        query_sha = hashlib.sha256(query.encode()).hexdigest()
        queries[query_sha] = query
        trace["query_sha256"] = query_sha
        return trace
    for name, answer, row, phase, origin in fixtures:
        obs, todo, strategy = input_context(row)
        a = pipeline(answer, False, obs, todo, strategy, phase, sha=BASELINE)
        b = pipeline(answer, True, obs, todo, strategy, phase)
        expected = original(answer)
        if b["valid"]:
            assert b["parsed"] == expected
            assert [x["action"] for x in b["executor_queue"]] == expected
        else:
            assert a["executor_queue"] == b["executor_queue"]
        result["cases"].append(dict(name=name, origin=origin, raw_answer=answer,
                                    A=compact(a), B=compact(b), archive_expanded=expected))
    answer = "- supplydepot x 3\n- barracks x 4"
    a = pipeline(answer, False, todo=dict(supplydepot=3, barracks=4, marine=8), phase=1, sha=BASELINE)
    b = pipeline(answer, True, todo=dict(supplydepot=3, barracks=4, marine=8), phase=1)
    assert b["parsed"] == original(answer)
    assert [x["action"] for x in b["executor_queue"]] == b["parsed"] + ["marine"]
    result["cases"].append(dict(name="opening_guard", origin="synthetic guard boundary", raw_answer=answer,
                                A=compact(a), B=compact(b), archive_expanded=original(answer)))
    result["queries"] = queries
    result["assertions_passed"] = True
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
