import sys,json,re,hashlib,collections,math
from pathlib import Path
code=Path("/root/new-scii-spb-fsdp-20260910/code-90b3fdc")
sys.path.insert(0,str(code/"scripts"))
from run_record import metric_rows
base=Path("/mnt/data0601/yangjinyue/new_SCII/experiments/20260910-spb-fsdp-layer-wrap-v1/runs")
specs=[
(6,0,"20260910-spb-fsdp-layer-wrap-v1-spb2-l6-g2-opening-quota-rl96-8gpu-s0-r1"),
(7,0,"20260910-spb-fsdp-layer-wrap-v1-spb2-l7-g2-opening-quota-rl4-4gpu-s0-r1"),
(7,4,"20260910-spb-fsdp-layer-wrap-v1-spb2-l7-g2-opening-quota-resume4to96-4gpu-s0-r1")]
for level,start,runid in specs:
 p=base/runid
 rows=metric_rows(p/"console.log")
 if not rows:continue
 last=max(map(int,rows));firstlog=min(map(int,rows))
 raw=[json.loads(l) for l in (p/"prompt_response.jsonl").read_text().splitlines()]
 out=[json.loads(l) for l in (p/"episode_outcomes.jsonl").read_text().splitlines()]
 seed={r["reset_index"]:r["seeds"][0] for r in map(json.loads,(p/"training_seeds.jsonl").read_text().splitlines())}
 cfg=json.loads((p/"effective_config.json").read_text());m=json.loads((p/"manifest.json").read_text())
 print("RUN",runid,"SHA",m["source_commit"],"LAST",last)
 print("SAMPLING_OVERRIDES",[x for x in cfg["command"] if re.search(r"(do_sample|temperature|top_p|top_k|capture_log_probs|reward|entropy_coeff|seed|REWARD|SHAPING|history_length|sc2_memory_mode)=",x) and not any(t in x for t in ["PYTHONPATH","LOG_PATH"])])
 first=min(r["step_index"] for r in raw)
 for step in range(firstlog,last+1):
  ends={r["env_index"]:r for r in out if r["reset_index"]==step-1}
  if len(ends)!=8:continue
  lo=max([r["step_index"] for r in out if r["reset_index"]==step-2],default=first-1)+1
  traj=[]
  for e in range(8):
   rr=[r for r in raw if r["env_idx"]==e and lo<=r["step_index"]<=ends[e]["step_index"]]
   pol=[r for r in rr if r["policy_action_applied"]]
   traj.append({"e":e,"result":ends[e]["game_result"],"reward":sum(r["reward"] for r in rr),"rows":pol})
  opening=[t["rows"][0] for t in traj]; ph=[hashlib.sha256(r["prompt"].encode()).hexdigest()[:12] for r in opening]
  a=collections.Counter(r["llm_response"] for r in opening)
  stat=rows.get(step,rows.get(str(step)))
  def sig(r):return tuple(x.strip().lower() for x in r["llm_response"].splitlines() if x.strip().startswith("-"))
  print("STEP",json.dumps(dict(level=level,step=step,seed=seed[step-1],wins=sum(t["result"]=="Victory" for t in traj),rewards=[round(t["reward"],6) for t in traj],opening_prompt_hashes=dict(collections.Counter(ph)),opening_answer_counts=[{"answer":k,"n":v} for k,v in a.most_common()],opening_bullet_unique=len(set(sig(r) for r in opening)),trajectory_unique=len(set(tuple(r["llm_response"] for r in t["rows"]) for t in traj)),lens=[len(t["rows"]) for t in traj],entropy=stat.get("actor/entropy_loss"),adv_min=stat.get("critic/advantages/min"),adv_max=stat.get("critic/advantages/max"),response_mean=stat.get("response_length/mean"))))
  # Selected concrete comparisons only; full prompt itself is not repeated.
  if (level==6 and step in [1,5,7,14,18,20]) or (level==7 and step in [1,4,5,6,7,18,20]):
   print("TRAJECTORIES",json.dumps({"level":level,"step":step,"examples":[{"env":t["e"],"reward":t["reward"],"result":t["result"],"plans":[{"prompt_sha":hashlib.sha256(r["prompt"].encode()).hexdigest()[:12],"answer":r["llm_response"],"valid":r["is_action_valid"],"failure":r["failure_reason"],"reward":r["reward"],"time":r["time"]} for r in t["rows"]]} for t in traj]},ensure_ascii=False))
