"""Read-only Level6 log audit; CPU scalar GiGPO replay, never model loading."""
import argparse
import ast
from collections import Counter, defaultdict
import contextlib
from difflib import SequenceMatcher
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence
import uuid

PREFIX = "20260910-spb-fsdp-layer-wrap-v1-"
SPECS = (
    ("parent", "spb2-l6-g2-opening-quota-rl96-8gpu-s0-r1", 0, 63,
     "6a1657f7e045dcf998495a3d0adfb331572bf60d", 0.95),
    ("A", "spb2-l6-g2-gamma0.95-s4plus8-8gpu-s0-r2", 4, 8,
     "5ba3f2111dc5bd576d1e67c7da5fdeaffba9f24f", 0.95),
    ("B", "spb2-l6-g2-gamma1.0-s4plus8-8gpu-s0-r2", 4, 8,
     "5ba3f2111dc5bd576d1e67c7da5fdeaffba9f24f", 1.0),
)


def jsonl(path):
    return [dict(json.loads(line), source_line=i) for i, line in
            enumerate(path.read_text().splitlines(), 1)]


def bullets(answer):
    # Text-normalized ordered bullets, not a claim of semantic equivalence.
    return tuple(line.strip().lower() for line in answer.splitlines()
                 if line.strip().startswith("-"))


def main(root, code):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    source = (code / "gigpo/core_gigpo.py").read_bytes()
    source_sha = hashlib.sha256(source).hexdigest()
    assert source_sha == "f135bd272518b49a58762da3474e2d4bc90906b053f1ae820d8fb05c2064148b"
    scope = dict(np=np, torch=torch, Counter=Counter, defaultdict=defaultdict,
                 uuid=uuid, SequenceMatcher=SequenceMatcher, Sequence=Sequence,
                 List=List, Dict=Dict, Any=Any, DataProto=object)
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    exec(compile(tree, str(code / "gigpo/core_gigpo.py"), "exec"), scope)
    sys.path.insert(0, str(code / "scripts"))
    from run_record import metric_rows
    sys.path.insert(0, str(code))
    from agent_system.sc2_episode import policy_transitions
    outputs, configs = {}, {}
    for label, suffix, start, count, sha, gamma in SPECS:
        folder = root / (PREFIX + suffix)
        manifest = json.loads((folder / "manifest.json").read_text())
        result = json.loads((folder / "result.json").read_text())
        assert manifest["source_commit"] == sha and not manifest["dirty"]
        assert manifest["world_size"] == 8 and manifest["mode"] == "training"
        assert result == json.loads((folder / ".done").read_text())
        assert result["completed_steps"] == count
        console_sha = hashlib.sha256((folder / "console.log").read_bytes()).hexdigest()
        assert console_sha == result["console_sha256"]
        cfg = json.loads((folder / "effective_config.json").read_text())
        configs[label] = cfg
        raw = jsonl(folder / "prompt_response.jsonl")
        outcomes = jsonl(folder / "episode_outcomes.jsonl")
        seeds = {r["reset_index"]:r["seeds"] for r in jsonl(folder / "training_seeds.jsonl")}
        metrics = metric_rows(folder / "console.log")
        batches, examples, observed_supply = [], [], []
        feedback_queries = 0
        pairing = None
        if start:
            pairing = json.loads((folder / "checkpoints/gamma_pairing.json").read_text())
            assert pairing["loaded_global_step"] == 4 and pairing["model_optimizer_load_returned"]
        for reset in range(count):
            step = start + reset + 1
            ends = {r["env_index"]:r for r in outcomes if r["reset_index"] == reset}
            assert len(ends) == 8
            low = max([r["step_index"] for r in outcomes if r["reset_index"] == reset-1],
                      default=min(r["step_index"] for r in raw)-1) + 1
            trajectories, flat, rewards = [], [], []
            for env in range(8):
                rows = [r for r in raw if r["env_idx"] == env and low <= r["step_index"] <= ends[env]["step_index"]]
                policy = policy_transitions([dict(r, active_masks=True,
                    rewards=r["reward"], environment_done=r["done"]) for r in rows])
                assert len(policy) == ends[env]["policy_decision_count"]
                assert ends[env]["terminated"] and not ends[env]["truncated"]
                rewards.append(sum(r["reward"] for r in rows))
                assert abs(sum(r["rewards"] for r in policy)-rewards[-1]) < 1e-5
                trajectories.append(policy)
                for ordinal, row in enumerate(policy):
                    flat.append(dict(row, traj=str(env), ordinal=ordinal,
                                     episode_reward=rewards[-1]))
            n = len(flat)
            assert len(seeds[reset]) == 8 and len(set(seeds[reset])) == 1
            for row in flat:
                obs = row["prompt"].split("Current observations are:\n", 1)[1].split("The remaining BO list", 1)[0]
                match = re.search(r"(\d+) remains unused", obs)
                assert match
                observed_supply.append(int(match[1]))
                feedback_queries += "Execution feedback for the previous segment" in row["prompt"]
            padded = math.ceil(n/8)*8
            invalid = sum(not r["is_action_valid"] for r in flat)
            m = metrics[step]
            assert abs(sum(rewards)/8 - m["episode/reward/mean"]) < 0.001
            returns = scope["compute_step_discounted_returns"](SimpleNamespace(
                batch={"input_ids":torch.zeros(n, 1, dtype=torch.long)},
                non_tensor_batch={"rewards":np.array([r["rewards"] for r in flat], np.float32),
                    "traj_uid":np.array([r["traj"] for r in flat], object),
                    "active_masks":np.ones(n)}), gamma)
            replay = {"status":"not_exact: invalid penalties or unlogged copy/order"}
            if not invalid and n == padded:
                with contextlib.redirect_stdout(io.StringIO()):
                    joint, _ = scope["compute_gigpo_outcome_advantage"](
                        torch.tensor([[r["episode_reward"]] for r in flat], dtype=torch.float32),
                        returns, torch.ones(n, 1),
                        np.array([r["prompt"] for r in flat], object),
                        np.array(["group"]*n, object),
                        np.array([r["traj"] for r in flat], object), mode="mean_norm")
                replay = dict(status="CPU scalar; valid, no copied padding",
                              min=joint.min().item(), max=joint.max().item(),
                              absmax=joint.abs().max().item())
            opening = Counter(bullets(t[0]["llm_response"]) for t in trajectories)
            sequence = Counter(tuple(bullets(r["llm_response"]) for r in t) for t in trajectories)
            batch = dict(global_step=step, seed=seeds[reset][0],
                wins=sum(r["game_result"] == "Victory" for r in ends.values()),
                rewards=rewards, calls=[len(t) for t in trajectories],
                decisions=n, invalid=invalid, copies=padded-n,
                inferred_optimizer_updates=padded//8,
                opening_bullet_variants=len(opening), trajectory_bullet_variants=len(sequence),
                opening_modes=[dict(bullets=k, count=v) for k,v in opening.most_common()],
                advantage_logged=[m["critic/advantages/min"], m["critic/advantages/max"]],
                cpu_joint=replay, grad_norm=m["actor/grad_norm"], entropy=m["actor/entropy_loss"],
                pg_loss=m["actor/pg_loss"], ppo_k1=m["actor/ppo_kl"], reference_kl=m["actor/kl_loss"],
                clipfrac=m["actor/pg_clipfrac"])
            batches.append(batch)
            if label == "parent" and step in (16,17,18,20):
                examples.append(dict(global_step=step, trajectories=[dict(env=e,
                    result=ends[e]["game_result"], reward=rewards[e],
                    answers=[dict(source_line=r["source_line"], answer=r["llm_response"],
                        prompt_sha256=hashlib.sha256(r["prompt"].encode()).hexdigest(),
                        reward=r["rewards"], valid=r["is_action_valid"]) for r in t])
                    for e,t in enumerate(trajectories) if e in (0,1)]))
        artifacts = {name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in
            ("manifest.json", "effective_config.json", "result.json", "prompt_response.jsonl",
             "episode_outcomes.jsonl", "training_seeds.jsonl")}
        zero = [b for b in batches if b["advantage_logged"] == [0,0]]
        outputs[label] = dict(run_id=PREFIX+suffix, source_sha=sha,
            status=result["status"], exit_code=result["process_exit_code"], ended_at=result["ended_at"],
            games=count*8, wins=sum(b["wins"] for b in batches),
            mean_reward=sum(sum(b["rewards"]) for b in batches)/(count*8),
            decisions=sum(b["decisions"] for b in batches), invalid=sum(b["invalid"] for b in batches),
            zero_supply_queries=observed_supply.count(0), execution_feedback_queries=feedback_queries,
            checkpoint_pairing=pairing,
            logged_zero_batches=[b["global_step"] for b in zero],
            exact_replay_batches=sum("absmax" in b["cpu_joint"] for b in batches),
            cpu_exact_zero_batches=[b["global_step"] for b in batches if b["cpu_joint"].get("absmax") == 0],
            zero_batch_optimizer_updates=sum(b["inferred_optimizer_updates"] for b in zero),
            inferred_optimizer_updates=sum(b["inferred_optimizer_updates"] for b in batches),
            batches=batches, examples=examples, artifacts=artifacts, console_sha256=console_sha)
    assert outputs["A"]["games"] == outputs["B"]["games"] == 64
    a, b = configs["A"]["overrides"], configs["B"]["overrides"]
    outputs["config_difference_raw"] = {k:[a.get(k),b.get(k)] for k in sorted(a.keys()|b.keys()) if a.get(k)!=b.get(k)}
    def normalized(value, label):
        return str(value).replace(outputs[label]["run_id"], "RUN").replace(
            f"/dev/shm/sc2_gamma_{label}_0911_r2", "TMP")
    difference = {k:[a.get(k),b.get(k)] for k in a.keys()|b.keys()
                  if normalized(a.get(k), "A") != normalized(b.get(k), "B")}
    assert difference == {"algorithm.gamma":["0.95", "1.0"]}
    outputs["normalized_method_difference"] = difference
    assert outputs["A"]["checkpoint_pairing"] == outputs["B"]["checkpoint_pairing"]
    assert [r["seed"] for r in outputs["A"]["batches"]] == [r["seed"] for r in outputs["B"]["batches"]]
    outputs["scope"] = "Level6 only; scalar replay, not saved PPO tensors or parameter gradients"
    outputs["gigpo_sha256"] = source_sha
    outputs["cuda_initialized"] = torch.cuda.is_initialized()
    print(json.dumps(outputs, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--code", type=Path, required=True)
    args = parser.parse_args()
    main(args.runs_root, args.code)
