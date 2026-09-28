import sys,ast,json,collections,uuid,contextlib,io,itertools,hashlib,re
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence,List,Dict,Any
from difflib import SequenceMatcher
code=Path("/root/new-scii-spb-fsdp-20260910/code-90b3fdc")
sys.path.insert(0,"/root/new-scii-spb2-level5-control/python")
import numpy as np,torch
torch.set_num_threads(1);assert not torch.cuda.is_initialized()
ns=dict(np=np,torch=torch,defaultdict=collections.defaultdict,Counter=collections.Counter,uuid=uuid,SequenceMatcher=SequenceMatcher,Sequence=Sequence,List=List,Dict=Dict,Any=Any,DataProto=object)
tree=ast.parse((code/"gigpo/core_gigpo.py").read_text());tree.body=[x for x in tree.body if isinstance(x,ast.FunctionDef)]
exec(compile(tree,"frozen_gigpo","exec"),ns)
base=Path("/mnt/data0601/yangjinyue/new_SCII/experiments/20260910-spb-fsdp-layer-wrap-v1/runs")
p=base/"20260910-spb-fsdp-layer-wrap-v1-spb2-l7-g2-opening-quota-resume4to96-4gpu-s0-r1"
raw=[json.loads(l) for l in (p/"prompt_response.jsonl").read_text().splitlines()]
out=[json.loads(l) for l in (p/"episode_outcomes.jsonl").read_text().splitlines()]
end={r["env_index"]:r for r in out if r["reset_index"]==4};lo=min(r["step_index"] for r in raw)
flat=[]
for e in range(8):
 rr=[r for r in raw if r["env_idx"]==e and lo<=r["step_index"]<=end[e]["step_index"]]
 assert all(r["policy_action_applied"] and r["is_action_valid"] for r in rr)
 assert abs(sum(r["reward"] for r in rr)-20.5)<1e-7
 for ordinal,r in enumerate(rr):flat.append(dict(r,traj=str(e),ordinal=ordinal))
n=len(flat);assert n==18
traj=np.array([r["traj"] for r in flat],object);anc=np.array([r["prompt"] for r in flat],object)
b=SimpleNamespace(batch={"input_ids":torch.zeros(n,1,dtype=torch.long)},non_tensor_batch={"rewards":np.array([r["reward"] for r in flat],np.float32),"traj_uid":traj,"active_masks":np.ones(n)})
large=next(i for i,r in enumerate(flat) if r["traj"]=="0" and r["ordinal"]==1)
small=next(i for i,r in enumerate(flat) if r["traj"]=="4" and r["ordinal"]==1)
for gamma in [0.95,1.0]:
 ret=ns["compute_step_discounted_returns"](b,gamma)
 results=[]
 for dup in itertools.combinations(range(n),2):
  ix=np.array(list(range(n))+list(dup));m=len(ix);mask=torch.ones(m,1);uid=np.array(["g"]*m,object)
  ep=ns["episode_norm_reward"](torch.full((m,1),20.5),mask,uid,traj[ix],remove_std=True)
  with contextlib.redirect_stdout(io.StringIO()):groups=ns["build_step_group"](anc[ix],uid,False)
  st=ns["step_norm_reward"](ret[ix],mask,groups,remove_std=True)
  joint=ep+st
  results.append((joint[large].item(),joint[small].item(),joint.abs().max().item()))
 a=np.array(results)
 print("GAMMA_COUNTERFACTUAL",json.dumps(dict(gamma=gamma,source="L7 step5 real rewards, same prompt, all153 legal 2-copy fillings",large_return=ret[large].item(),small_return=ret[small].item(),large_joint_range=[a[:,0].min(),a[:,0].max()],small_joint_range=[a[:,1].min(),a[:,1].max()],global_joint_absmax=a[:,2].max(),fillings=len(results))))
# Inspect existing lightweight states and exact same-prompt opening switch.
q=base/"20260910-spb-fsdp-layer-wrap-v1-spb2-l6-g2-opening-quota-rl96-8gpu-s0-r1"
qr=[json.loads(l) for l in (q/"prompt_response.jsonl").read_text().splitlines()]
qo=[json.loads(l) for l in (q/"episode_outcomes.jsonl").read_text().splitlines()]
lo=max(r["step_index"] for r in qo if r["reset_index"]==15)+1
ed={r["env_index"]:r for r in qo if r["reset_index"]==16}
for e in range(8):
 rr=[r for r in qr if r["env_idx"]==e and lo<=r["step_index"]<=ed[e]["step_index"]]
 print("L6_STEP17",json.dumps(dict(env=e,result=ed[e]["game_result"],reward=sum(r["reward"] for r in rr),answers=[r["llm_response"] for r in rr[:2]],first_prompt_sha=hashlib.sha256(rr[0]["prompt"].encode()).hexdigest(),calls=ed[e]["policy_decision_count"]),ensure_ascii=False))
for run,step in [(q,20),(p,16)]:
 config=json.loads((run/"checkpoints"/f"global_step_{step}"/"actor/config.json").read_text())
 print("CHECKPOINT_DROPOUT",run.name,config.get("swa_dropout"))
 # Output matching key lines only, for resolved config.
 log=(run/"console.log").read_text(errors="replace")[:120000]
 for key in ["do_sample","capture_log_probs","swa_dropout","enable_similarity"]:
  print("RESOLVED",key,[l.strip() for l in log.splitlines() if key in l and len(l)<300][:5])
print("CPU_ONLY",not torch.cuda.is_initialized())
