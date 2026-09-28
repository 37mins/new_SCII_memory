"""Fixed failure-query model/example comparisons; no game or PPO."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "experiments/records/20260909-bo-segment-protocol-v1"))
from fixed_prompt_ab import load_config, now, sha
from test_r6_format_reward import planner_scope, strict_validator

SOURCE_SHA = "d155c96eaccf9aa809544945b387f0e17e1ca5bd"
LOG_SHA = "15161b5dc3a18da54a4d7281be3a31d5522dd93bde99bdb22c8cc0915e7dd292"
PROMPT_SHAS = ["eeb0089b61d15439464968c80036a8f1964a7a133507685e55dd08625630d0b3",
               "45f6664d8ca8e4079b360c5ac69fcba97ddf4745b19d7a30e1067ca00e3f849f"]


def load_cases(path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != LOG_SHA:
        raise ValueError("Not the completed r3 B response log")
    cases = []
    for line, text in enumerate(path.read_text().splitlines(), 1):
        row = json.loads(text)
        if row["env_idx"] != 0 or row["step_index"] not in (2, 3):
            continue
        assert row["policy_action_applied"] and row["semantic_action_evaluable"]
        prompt = row["prompt"]
        index = len(cases)
        assert sha(prompt) == PROMPT_SHAS[index]
        assert "- supplydepot x 8\n" in prompt and "0 remains unused" in prompt
        cases.append(dict(index=index, source_line=line, env=0, game_seed=52620,
                          step_index=row["step_index"], prompt=prompt,
                          prompt_sha256=sha(prompt), historical_answer=row["llm_response"],
                          sampling_seeds=[11000 + 4 * index + r for r in range(4)]))
    assert len(cases) == 2
    return cases


def score(prompt, answer):
    valid, reason = strict_validator()(prompt, answer)
    todo = prompt.split("The remaining BO list in this phase:\n", 1)[1].split("Use only exact canonical names", 1)[0]
    names = [re.sub(r"\s+x\s+\d+$", "", line[2:]) for line in todo.splitlines() if line.startswith("- ")]
    queue = planner_scope()["parse_build_order"](
        answer, choice_map={str(i):name for i, name in enumerate(names)},
        strict_names_only=True, original_segment=True)
    depot = queue.index("supplydepot") if "supplydepot" in queue else None
    blocked = next((i for i, name in enumerate(queue) if name in ("marine", "scv")), None)
    return dict(valid=bool(valid), reason=reason, parsed_queue=queue,
                depot_before_supply_consumer=bool(valid and depot is not None and
                                                  (blocked is None or depot < blocked)),
                first_action=queue[0] if queue else None,
                depot_first=bool(valid and queue and queue[0] == "supplydepot"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--example-order", action="store_true",
                        help="A4 only; same failure input, swap example Marine/Depot order")
    modes.add_argument("--paired-a4-run", type=Path,
                       help="Generate only original SpB; pair with this completed A4 arm")
    modes.add_argument("--failure-panel", type=Path,
                       help="Original SpB versus A4 on three frozen inputs, eight paired draws each")
    args = parser.parse_args()
    import torch
    import torch.distributed as dist
    import numpy as np
    from omegaconf import OmegaConf
    from verl import DataProto
    from verl.workers.fsdp_workers import ActorRolloutRefWorker
    from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector

    assert int(os.environ["WORLD_SIZE"]) == 4
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_num_threads(1)
    source_sha, log_sha = SOURCE_SHA, LOG_SHA
    case_loader = load_cases
    a4_input = args.example_order or args.paired_a4_run is not None
    if a4_input:
        import example_order_probe as probe
        source_sha, log_sha, case_loader = probe.SOURCE_SHA, probe.LOG_SHA, probe.load_cases
    if args.failure_panel:
        sys.path.insert(0, str(ROOT / "experiments/records/20260916-combat-result-reward-v1"))
        from failure_panel_probe import load_panel, summarize_panel
        panel, cases = load_panel(args.failure_panel)
        source_sha = os.environ["SOURCE_SHA"]
        log_sha = panel["artifact_sha256"]["prompt_response.jsonl"]
    elif args.paired_a4_run:
        import original_a4_probe
        cases, reference = original_a4_probe.prepare(args.source_run, args.paired_a4_run)
    else:
        cases = case_loader(args.source_run / "prompt_response.jsonl")
    source_manifest = json.loads((args.source_run / "manifest.json").read_text())
    assert source_manifest["source_commit"] == source_sha
    config = load_config(args.source_run)
    # Match trainer's initialization assignment, with no trainer instantiated.
    config.actor_rollout_ref.actor.optim.total_training_steps = config.trainer.total_training_steps
    if args.failure_panel:
        # This source is a training dry-run, whose unused validation defaults
        # are greedy. The probe uses the actual approved rollout sampling.
        for name in ("temperature", "top_p", "top_k", "do_sample"):
            config.actor_rollout_ref.rollout.val_kwargs[name] = config.actor_rollout_ref.rollout[name]
    if args.paired_a4_run:
        old_config = json.loads((args.paired_a4_run / "effective_config.json").read_text())
        assert OmegaConf.to_container(config, resolve=True) == old_config, "Paired generation config changed"
    assert config.actor_rollout_ref.rollout.response_length == 192
    assert config.actor_rollout_ref.rollout.fsdp_layer_wrap
    assert config.actor_rollout_ref.rollout.val_kwargs.temperature == 0.4
    if a4_input:
        checkpoint = Path(config.trainer.resume_from_path)
        assert checkpoint.name == "global_step_4" and checkpoint.parent.parent.name == probe.CHECKPOINT_RUN
        assert source_manifest["resume_from_path"] == str(checkpoint)
        assert config.env.history_length == 0 and config.env.sc2_memory_mode == "off"
        assert config.actor_rollout_ref.rollout.val_kwargs.top_p == 0.9
        assert config.actor_rollout_ref.rollout.val_kwargs.top_k == 0
        assert config.actor_rollout_ref.rollout.val_kwargs.do_sample
    if args.failure_panel:
        assert Path(config.trainer.resume_from_path).name == "global_step_4"
        assert config.env.history_length == 0 and config.env.sc2_memory_mode == "off"
        assert config.actor_rollout_ref.rollout.val_kwargs.top_p == 0.9
        assert config.actor_rollout_ref.rollout.val_kwargs.top_k == 0
        assert config.actor_rollout_ref.rollout.val_kwargs.do_sample
    stream = (args.output / f"rank-{rank}.jsonl").open("x")

    def emit(event, **values):
        stream.write(json.dumps(dict(event=event, rank=rank, time=now(), **values), ensure_ascii=False) + "\n")
        stream.flush()

    if rank == 0:
        if args.paired_a4_run:
            (args.output / "paired_a4_reference.json").write_text(json.dumps(reference, ensure_ascii=False, indent=2))
        if args.failure_panel:
            (args.output / "failure-panel.json").write_text(json.dumps(panel, ensure_ascii=False, indent=2))
        (args.output / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2))
        (args.output / "effective_config.json").write_text(json.dumps(OmegaConf.to_container(config, resolve=True), indent=2))
        (args.output / "manifest.json").write_text(json.dumps(dict(
            source_commit=os.environ["SOURCE_SHA"], source_run=str(args.source_run),
            source_runtime_sha=source_sha, source_log_sha256=log_sha,
            model_manifest_sha256=source_manifest["model_manifest_sha256"],
            checkpoint=config.trainer.resume_from_path, original_model=config.actor_rollout_ref.model.path,
            status="running", started_at=now(), world_size=4, gpus=os.environ["CUDA_VISIBLE_DEVICES"],
            max_wall_seconds=1800 if args.failure_panel else 900 if args.paired_a4_run else 1200,
            games=0, optimizer_updates=0, answers_per_arm=24 if args.failure_panel else 8,
            checkpoint_loaded_in_this_run=bool(not args.paired_a4_run),
            paired_a4_run=str(args.paired_a4_run) if args.paired_a4_run else None,
            contexts=("three historical failure queries; eight paired draws each, not games" if args.failure_panel
                      else "one 05:49 failure query; eight paired draws, not eight games" if a4_input
                      else "two sequential failures in env0, not independent games"),
            sampling=("12000 + 8*context_index + 4*draw_group + rank, reset before each model" if args.failure_panel
                      else "12000 + 4*draw_group + rank, reset before each arm" if a4_input
                      else "11000 + 4*context_index + rank, reset before each weight/case"),
            changed=("example Marine2/Depot2 line order only" if args.example_order else "model weights"),
            unchanged="tokenizer, sampling, FP32 parameters, BF16 autocast, cache, FSDP, dropout0",
            reuse="fixed_prompt_ab.load_config, TrajectoryCollector, ActorRolloutRefWorker and full checkpoint loader"), indent=2))

    worker = ActorRolloutRefWorker(config.actor_rollout_ref, role="actor_rollout")
    worker.init_model()
    assert worker.actor_model_config.model_type == "sse_swa_moba"
    assert worker.actor_model_config.swa_dropout == 0
    assert next(worker.actor_module_fsdp.parameters()).dtype == torch.float32
    assert len(worker.actor_optimizer.state) == 0
    if args.example_order:
        worker.load_checkpoint(str(Path(config.trainer.resume_from_path) / "actor"),
                               del_local_after_load=False)
    collector = TrajectoryCollector(config, worker.tokenizer)
    dummy = SimpleNamespace(non_tensor_batch=dict(raw_prompt=[None], data_source=["sc2"]))

    arms = (("original", "A4") if args.failure_panel else ("original",) if args.paired_a4_run
            else (("A", "B") if args.example_order else ("original", "step24")))
    for arm in arms:
        if arm in ("step24", "A4"):
            worker.load_checkpoint(str(Path(config.trainer.resume_from_path) / "actor"),
                                   del_local_after_load=False)
        counters = sorted({int(s["step"].item()) for s in worker.actor_optimizer.state.values()})
        assert bool(counters) == (args.example_order or arm in ("step24", "A4"))
        if args.failure_panel and arm == "A4":
            assert counters == [15], "Expected original A4 Adam counters, no new updates"
        emit("model_ready", arm=arm, checkpoint_load_returned=args.example_order or arm in ("step24", "A4"),
             optimizer_counters=counters, swa_dropout=0, dtype="fp32", cache=True)
        for case in cases:
            prompt = case["prompts"][arm] if args.example_order else case["prompt"]
            sample = collector.preprocess_single_sample(0, dummy, dict(text=[prompt], image=None, anchor=[prompt]))
            rendered = worker.tokenizer.apply_chat_template([dict(role="user", content=prompt)],
                        tokenize=False, add_generation_prompt=True, **config.data.apply_chat_template_kwargs)
            full_ids = worker.tokenizer.encode(rendered, add_special_tokens=False)
            actual_ids = sample["input_ids"][sample["attention_mask"].bool()].tolist()
            assert full_ids == actual_ids, "Saved prompt was truncated or formatted differently"
            batch = DataProto.from_dict(tensors={k:sample[k].unsqueeze(0) for k in
                ("input_ids", "attention_mask", "position_ids")}, meta_info=dict(validate=True, do_sample=True))
            seed = case["sampling_seeds"][rank]
            torch.manual_seed(seed)
            np.random.seed(seed)
            rng_sha = hashlib.sha256(bytes(torch.cuda.get_rng_state().tolist())).hexdigest()
            if args.paired_a4_run:
                old_start = reference["starts"][seed]
                assert old_start["input_token_sha256"] == sha(json.dumps(actual_ids))
                assert old_start["cuda_rng_sha256"] == rng_sha
            emit("generation_start", arm=arm, case=case["index"], seed=seed,
                 input_token_sha256=sha(json.dumps(actual_ids)), input_tokens=len(actual_ids),
                 cuda_rng_sha256=rng_sha)
            result = worker.generate_sequences(batch)
            ids = result.batch["responses"][0].tolist()
            eos = worker.tokenizer.eos_token_id
            response = worker.tokenizer.decode(ids, skip_special_tokens=True)
            emit("generation", arm=arm, case=case["index"], seed=seed,
                 raw_answer=response, token_ids=ids, prompt_sha256=sha(prompt),
                 eos_seen=eos in ids, token_count=ids.index(eos)+1 if eos in ids else len(ids),
                 **score(prompt, response))
        assert counters == sorted({int(s["step"].item()) for s in worker.actor_optimizer.state.values()})
        emit("arm_complete", arm=arm, optimizer_updates=0, optimizer_counters=counters)
    dist.barrier()
    if rank == 0:
        rows = [json.loads(l) for r in range(4) for l in (args.output / f"rank-{r}.jsonl").read_text().splitlines()]
        answers = [r for r in rows if r["event"] == "generation"]
        assert len(answers) == (24 if args.failure_panel else 8) * len(arms)
        summary = dict(status="completed", ended_at=now(), games=0, optimizer_updates=0, arms={})
        for arm in arms:
            selected = [r for r in answers if r["arm"] == arm]
            summary["arms"][arm] = {key:sum(bool(r[key]) for r in selected) for key in
                ("valid", "depot_before_supply_consumer", "depot_first", "eos_seen")}
            summary["arms"][arm]["answers"] = len(selected)
        if args.paired_a4_run:
            summary["paired_a4"] = original_a4_probe.compare(answers, reference)
        if args.failure_panel:
            summary["failure_panel"] = summarize_panel(rows, panel)
        (args.output / "result.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)
    dist.barrier()
    stream.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
