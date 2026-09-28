# v1524 micro on canonical Level 6 RL

This experiment tests whether the archived `v1524-frozen-20260725` micro
behavior improves the canonical Qwen Level 6 RL system without changing its
Planner, Strategist, Commander, Manager, build order, reward, or GIGPO
configuration.

The baseline is
`20260819-qwen-spb-canonical-level6-rl-v1-qwen-base-l6-full96-s0-r1`.
The only behavior change is `SC2_ENABLE_NEWEXEC_MICRO=0 -> 1`. The frozen
`ScriptPlayer` remains byte-identical at SHA-256
`76354ff908b47f8c400e2cf279a3af5f49e98a3da6afb53018143b579f28c6da`;
the earlier five-BO gate established that its micro methods match the archived
v1524 source and that the switch executes real micro actions.

The first run is an eight-update smoke on Flat128 against the built-in Terran
`Harder` (`Level 6`) `Air` opponent, using eight GPUs and eight rollouts per
update. It starts from the official Qwen3-4B public base snapshot with no SFT
and no prior project RL. A complete step-8 checkpoint is required. The smoke
is used to check numerical health, action validity, reward, success rate, and
micro traces before any 96-step promotion.

## Runs

`20260829-v1524-micro-level6-rl-v1-qwen-base-l6-smoke8-s0-r1` started at
2026-08-30 00:03 CST on `sc2-a100-44`, GPUs 0-7. It runs from clean Git
commit `202884da58b8e9e276afae33b762c702a119545d`; the immutable deployment
archive SHA-256 is
`9c672ccbdfb8dfd925d52277560cffe6eecf1bd7f1ee261fd694494bef35eb35`.
The run root is
`runs/20260829-v1524-micro-level6-rl-v1/20260829-v1524-micro-level6-rl-v1-qwen-base-l6-smoke8-s0-r1`
on the shared filesystem. It completed all eight updates with exit status 0 and
a complete `global_step_8` checkpoint.

## Result

The smoke did not pass the promotion gate. `episode/success_rate` was 0 for
every update, corresponding to no successes in the 64 training rollouts. Mean
episode reward was -42.719 and did not improve from step 1 (-40.368) to step 8
(-43.768).

The mean valid-action ratio was 6.85%. It fell to 0.8%, 0.8%, and 2.3% in the
last three updates. The mean response clipping ratio was 81.84%, reaching 100%
in steps 7 and 8, so the later Planner responses all reached the 96-token
budget. Gradients and losses remained finite: the pre-clipping gradient norm
averaged 25.628 and peaked at 44.689. The training log contained no traceback,
NaN, Inf, or non-finite failure.

The `MULEMICRO_TRACE` stream contained 360 non-empty entries, including tank
siege/unsiege, target focus, retargeting, preservation, kiting, and focus-fire
events. The v1524 switch therefore changed real execution behavior as intended;
the failure is not caused by an inactive micro integration.

This result does not support promotion to 96 updates. The earlier fixed-BO gate
reached two wins, two ties, and one loss because it supplied executable build
orders directly. In this RL smoke, low action validity and severe response
truncation show that the dynamic Planner output remains the immediate bottleneck;
micro alone cannot compensate for it. A causal comparison of micro on versus
off would still require a paired run with otherwise identical code and seeds.
