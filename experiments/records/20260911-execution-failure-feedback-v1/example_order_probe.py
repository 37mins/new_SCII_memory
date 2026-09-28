"""One recorded supply-block failure: swap only the example's first two bullets."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path

SOURCE_SHA = "e3c0695adf0a2e36c97afba838ec3b508f000e34"
LOG_SHA = "b1729caf9394a50072feda8f6836ae0907eb19b0b65f2d56df363373a9d61bf1"
PROMPT_SHA = "18e7797126e1af1d9b48a27eb82293bfa457e9be081c72ba1647cef870955627"
CHECKPOINT_RUN = "20260911-execution-failure-feedback-v1-spb2-original-l6-g2-A-rl4-4gpu-s0-r2"
MARKER = "The example below is generated from that same current list. Follow its format:\n\n"
EXAMPLE_A = "<answer>\nHere are the Build Orders with attack instructions:\n- marine x 2\n- supplydepot x 2\n- barracks x 2\n</answer>"
EXAMPLE_B = EXAMPLE_A.replace("- marine x 2\n- supplydepot x 2", "- supplydepot x 2\n- marine x 2")


def candidate(prompt):
    if prompt.count(MARKER + EXAMPLE_A) != 1:
        raise ValueError("Expected the exact historical example once")
    return prompt.replace(MARKER + EXAMPLE_A, MARKER + EXAMPLE_B, 1)


def load_cases(path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != LOG_SHA:
        raise ValueError("Expected the completed firstattack40 response log")
    rows = [(i, json.loads(line)) for i, line in enumerate(path.read_text().splitlines(), 1)]
    matches = [(i, r) for i, r in rows if r["env_idx"] == 0 and r["step_index"] == 2]
    assert len(matches) == 1
    line, row = matches[0]
    assert row["policy_action_applied"] and row["semantic_action_evaluable"]
    prompt = row["prompt"]
    assert hashlib.sha256(prompt.encode()).hexdigest() == PROMPT_SHA
    prompts = dict(A=prompt, B=candidate(prompt))
    # Two four-rank draws, NOT two independent game contexts.
    return [dict(index=i, source_line=line, env=0, game_seed=42613, step_index=2,
                 prompt=prompt, prompts=prompts, prompt_sha256=PROMPT_SHA,
                 historical_answer=row["llm_response"], sampling_seeds=list(range(12000+4*i, 12004+4*i)))
            for i in range(2)]


def main():
    from fixed_feedback_compare import score
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.source_run / "manifest.json").read_text())
    assert manifest["source_commit"] == SOURCE_SHA
    case = load_cases(args.source_run / "prompt_response.jsonl")[0]
    for arm, prompt in case["prompts"].items():
        print(json.dumps(dict(arm=arm, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
              historical_answer_score=score(prompt, case["historical_answer"])), ensure_ascii=False))
    print("".join(difflib.unified_diff(case["prompts"]["A"].splitlines(True),
                                      case["prompts"]["B"].splitlines(True), fromfile="A", tofile="B")))
    print("CPU input/protocol check only; no tokenizer/model/game/PPO has run.")


if __name__ == "__main__":
    main()
