# v1524 micro on the canonical Level 6 system

This evaluation asks whether the archived `v1524-frozen` micro layer can run
inside the canonical Qwen Level 6 system without changing planning, strategy,
macro production, attack scheduling, or training state.

The frozen Level 6 `ScriptPlayer` (`76354ff9…`) and the local v1524 archive
(`8c1c4812…`) contain the same 72 methods except
`_select_attack_compatible_enemy_unit`. All micro methods, their state, and the
maintenance hook are identical. The Level 6 contract fixes
`SC2_ATTACK_TARGET_UNITS_FIRST=0`, so the differing target-selection method is
inactive. The deployment therefore keeps the frozen Level 6 source unchanged
and changes only `SC2_ENABLE_NEWEXEC_MICRO` from `0` to `1`.

Five fixed build orders are evaluated without training. Each run performs one
deterministic validation trajectory with the official Qwen3-4B checkpoint on
Flat128 against the built-in Terran `Harder` (`Level 6`) `Air` opponent. The
historical BO seeds are retained: g1/752, g2/757, g3/756, g4/781, and g5/755.
The Planner contract remains legacy, six bullets per segment, no thinking;
automatic Marine production and forced attacks remain disabled.

`verify_v1524_compat.py` proves the source relationship before SC2 starts.
`launcher.sh` runs one BO in validation-only mode and saves its effective
configuration, console log, and replay. `run_five_bos.sh` prepares isolated
runtime copies and schedules g1–g5 on five GPUs.

The archived v1524 file is provenance evidence, not deployed code. Its exact
July 25 server tree could not be re-read because the only surviving connection
helper disables SSH host-key checking. The source used here is the local
archive `script_player.v1524.no_merge_into_attacking.py`, SHA-256
`8c1c48126e444ecf604a7ac79989c5c9e8dd06eaca31a5e8786b0e5f16ae4e8c`.

## Runs

`20260829-v1524-micro-level6-fivebo-qwen-base-r1` is invalid. All five
processes stopped before loading the model because the Ray temporary path
exceeded Linux's Unix-domain socket limit. No GPU was occupied, SC2 did not
start, and no replay was produced. The retry shortens only the Ray temporary
path and uses a new run ID; the experimental contract is unchanged.

`20260829-v1524-micro-level6-fivebo-qwen-base-r2` completed on
`sc2-a100-42`. All five validation processes exited with status 0 and saved a
replay. The result was two victories, two ties, and one defeat (win rate 2/5;
non-loss rate 4/5):

| BO | Seed | Result | Validation score | Last logged game time | Replay SHA-256 |
|---|---:|---|---:|---:|---|
| g1 | 752 | Tie | -48.097 | 14:50 | `bf247fb35ebe33727d0de28f11affce2f1840841d82f977132f796e6a7330251` |
| g2 | 757 | Defeat | -45.676 | 11:17 | `32930a5ba17f0f5ccf82bb1f3cd23fabea8c72f4da5b1436b65072f719b4481e` |
| g3 | 756 | Victory | 25.500 | 12:36 | `86d577685366f070eca72aaf6f8da6d3e73577ced3a381d6893f9940c5417451` |
| g4 | 781 | Victory | 20.000 | 09:20 | `c07e96a662eff6ed3b72da16da29af6bdeca0f55cfb15226dd2dbfaab3ba3c3a` |
| g5 | 755 | Tie | -46.822 | 14:01 | `ba5a5a5328bc6a17cce7f4d31924dc8408c898ce5d0b9ee5841580418c89a89d` |

The micro trace confirms that the enabled layer executed real behavior rather
than only accepting a configuration flag. Across the runs it recorded target
focus, focus fire, kiting, dodge preservation, tank siege/unsiege, Liberator
mode changes, Ghost snipe, Banshee cloak, Thor mode selection, and Viking mode
selection. There were no `training/global_step` records and no checkpoint
files, confirming that the evaluation performed no optimization updates.

The frozen deployment is
`deployments/20260829-v1524-micro-level6-fivebo-9954254`, built from Git commit
`99542542df3c87ed834a19c2b8f510eb8c3e73b8`. Raw runs and replays are under
`runs/20260829-v1524-micro-level6-fivebo/` on the shared filesystem.

This run establishes compatibility and a five-BO gate result. It does not by
itself estimate the causal gain from v1524 micro because there is no paired
micro-disabled run on the same BOs and seeds. Such an A/B comparison is needed
before attributing the two victories or four non-losses to the micro layer.

`20260830-v1524-micro-level5-fivebo-qwen-base-r1` repeats the same five BOs,
model, historical seeds, and v1524 micro configuration on the `Hard` Level 5
opponent. The launcher change only parameterizes the existing difficulty name;
its default remains `Harder`, preserving the completed Level 6 run. The Level 5
run started at 2026-08-30 03:57 CST on `sc2-a100-44`, GPUs 0-4, without
training. It uses clean commit
`f37914c5e12d3db41f446e623350d37d18e8a1a2`; the immutable deployment archive
SHA-256 is
`5539b3fc297b4af3d2c8d1a79c08bdcf0ec5f35582f625708bb5993aaa1387f3`.
Raw results are under `runs/20260830-v1524-micro-level5-fivebo/` on the shared
filesystem. This run was stopped and marked invalid because using a different
environment seed for each BO confounds the requested cross-BO comparison. Its
partial outputs are retained and must not be reported as efficacy results.

`20260830-v1524-micro-level5-fivebo-qwen-base-s0-r2` is the corrected run. It
uses environment seed 0 for every one of g1-g5 and keeps all other Level 5
settings unchanged. The shared seed also matches the concurrent Level 5 RL
experiment. It started at 2026-08-30 04:07 CST on `sc2-a100-44`, GPUs 0-4,
without training. It uses clean commit
`3d7e410729ed7fdbf96e3d523bde4b8021df47fc`; the immutable deployment archive
SHA-256 is
`4fcefa5620652119d8daa87ec4c8172ffc7769ed0a08530ed75b9b9502e3c51b`.
It completed at 2026-08-30 12:18 CST. All five validation processes exited
with status 0 and saved a replay. The result was four victories and one defeat
(win and non-loss rate 4/5):

| BO | Seed | Result | Validation score | Replay SHA-256 |
|---|---:|---|---:|---|
| g1 | 0 | Defeat | -49.599 | `9c841af04231e455eaed38c67a6a939372875c0d3083ff8fb4d9b54eb610102a` |
| g2 | 0 | Victory | 20.000 | `77fed67ac7f0ae613af1de7562d363b8b3e6d8d0a3cb540305871add041d2642` |
| g3 | 0 | Victory | 23.486 | `c45cc301e0733b697467546116365975b6817a8f065e08d2e217ec768e916606` |
| g4 | 0 | Victory | 20.000 | `ccec8e90881090904619d8685d6b3871623b22ef8684fb082a9a9d6fedc3ed0d` |
| g5 | 0 | Victory | 20.000 | `e221bb719e2c4dec72e7cbba46aa04c4511d57b6fb52a15e716e7a01a9b37364` |

Because all BOs now use the same seed, this run removes the seed confound in
the invalid Level 5 run. It is still a one-trajectory-per-BO functional gate,
not a statistical estimate of BO quality or the causal gain from micro.

`20260830-v1524-micro-level5-fivebo-qwen-base-s0-oldmicro-r3` was intended as
the paired old-micro control but is invalid. Its launch metadata expanded the
correct short commit `fecdb01` to an incorrect full SHA. The run was stopped
before completion and all partial outputs are excluded.

`20260830-v1524-micro-level5-fivebo-qwen-base-s0-oldmicro-r4` is the corrected
control. It repeats the completed seed-0 Level 5 run with the same model, five
BO files, evaluator, and runtime contract. The only behavioral change is
`SC2_ENABLE_NEWEXEC_MICRO=1 -> 0`; training remains disabled. It started at
2026-08-30 15:44 CST on `sc2-a100-44`, GPUs 0-4, from clean commit
`fecdb0160547b9e9dc7b04302a23b761b67ee521`. The immutable deployment is
`deployments/20260830-v1524-level5-fivebo-oldmicro-fecdb01`; its source archive
SHA-256 is
`d119d2337e2c43945b147ffbfcbb5a51237026007969d945be9d8de088317399`.
It completed at 2026-08-30 15:54 CST. All five processes exited with status 0
and saved a replay:

| BO | Seed | Old micro result | Old score | New micro result | New score |
|---|---:|---|---:|---|---:|
| g1 | 0 | Defeat | -50.000 | Defeat | -49.599 |
| g2 | 0 | Victory | 24.500 | Victory | 20.000 |
| g3 | 0 | Victory | 20.000 | Victory | 23.486 |
| g4 | 0 | Victory | 20.000 | Victory | 20.000 |
| g5 | 0 | Victory | 20.000 | Victory | 20.000 |

Both modes won 4/5 trajectories. Mean validation score was 6.900 for old
micro and 6.777 for new micro. This one-trajectory-per-BO gate shows no
observable outcome improvement from enabling v1524 micro. It establishes
functional parity, not statistical equivalence; causal efficacy requires
replicated paired seeds or training-curve comparison.
