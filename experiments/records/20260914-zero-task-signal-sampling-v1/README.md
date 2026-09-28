# 有上限的零任务信号组过滤

最新（2026-09-16）：46的v2对照A实际新增3步至global7，24局6胜，第四批因
空技能查询断言失败；B尚未启动，无新完整checkpoint。step6/7各0/8，组内
回答、回报趋同且仅有浮点残差信号；历史同种子也全败，不足以认定本次训练退化。
[运行及学习信号分析](node46-r2-analysis.md)与[两行边界修复](empty-abilities-fix.md)
分别记录。修复CPU通过、尚未实局验收；旧run和A4保留，未自动重试。

当前开发（2026-09-15）：用户批准收敛到 RL 信号供给；新增
[分组原始分数判定 v2](grouped-score-v2.md)，只替换下述 v1 的浮点严格非零判定。
v1 和既有场景训练/评测结果保留；v2 的 CPU 与 GPU 验收单独记录，不回写旧结论。

9月15日晚，[信号供给下一步](signal-supply-next.md)明确先验收v2，再单独验证场景选择；
已核对候选场景与共同评测种子分离。尚未接入动态采样，也未启动新GPU实验。
后续用户要求[限时等卡启动](signal-supply-next.md)：仅调度46四卡现有v2 A/B，
到原截止时间退出、只派发一次；监听运行不等于训练已经启动。

最新（2026-09-15 11:17）：[训练场景采样A/B](scene-training.md) 已收齐：A在43完成
新增4步至step8、32局29胜；B首试主机OOM保留，获批迁移47的r2于11:03验收完成，
新增4步、32局31胜、完整step8 checkpoint、退出0。B后3批task advantage精确零，
未达到持续提供任务信号的目标，不能视为RL学习问题已修复。两组训练种子不同，
尚无共同条件独评，不直接用胜率差判断学习收益。[B收尾证据](scene-B-r2-final.json)。

后续共同条件独评于9月15日12:05:09完成：47八卡分两组，A4→SceneB8与SceneA8
并行，共24局、均退出0、无PPO；SHA`8069b4f`。三组均5/8胜，同样三个seed失败；
平均整局reward为−5.72275/−5.728875/−5.728875，各16/16真实回答合格。
A8、B8与A4各有14/16回答逐字相同，未见本轮新增四步带来的评测收益。
12:08八卡显存均释放；未续96步。[收尾证据](scene-posteval-r1-final.json)。

随后只读核对[三批非零信号](nonzero-plan-credit.md)：A7确有同query下Marine42/8
对应5胜/3败及正负advantage；A6主要区分调用时序，B5同时涉及Marine/Depot差异。
训练后回答趋向42，但三个独评失败场景未改善；未发现新实现bug，未启动新实验。

用户已批准[SceneB8温度复核](scene-temperature.md)：47 GPU4–7、同失败seed，
T0.4/0.8各8局串行，共45分钟，无PPO。16:56仍无已确认可用卡组，47 SSH未连通，
尚未启动。冻结源码`8388860`已完成共享盘备份，SceneB8父checkpoint小文件核验通过；
没有以准备完成代替实际模型加载和游戏验收。

状态：2026-09-14，零组A/B均完成至global8、各32局7胜，B跳过0组；严格非零判定放行了两批5.72e-6残差，本候选未采用为有效修复，[收尾](pair-r1-final.json)。随后同一A4的[64局新场景纯评测](scene-screening.md)于22:51完成、55胜、全部退出0。CPU冻结函数重建：5/8组精确零，1组有胜负区分，另2组非零涉及调用长度/格式与回报差，[证据与范围](scene-signal.md)。仅一个新混合场景，未达到原“两新”门槛；未自动训练。原运行SHA与生产模块、阈值均未改。

## 问题与唯一变化

基线是 Coverage A/B 实际运行源码 `ce61865ac709fae44be22e705a8349720a19df52`，不是后来的阶段重规划候选。旧部署、原 r6 和历史 run 不改。开发分支 `codex/feature-zero-signal-sampling`。

现有学习信号审计指出有整批 GiGPO task advantage 为精确零的情况。候选在 old/ref 概率计算及 PPO 之前，用原 `EpisodeRewardManager`、invalid-action penalty 和 GiGPO 函数检查真实回答。有效 token 的 advantage **全部精确为零**才跳过整组；只要一条非零就保留整组，包含败局。跳过时不调用 actor 更新，因此没有这批的 PPO、KL、entropy、Adam 或 scheduler 更新，也不增加 global step。

这是 DAPO 动态采样思想在现有 SC2/GiGPO 上的适配，不是完整 DAPO。DAPO 2025 年论文 §3.2 过滤无组间奖励差的组；[论文](https://arxiv.org/html/2503.14476v2#S3.SS2)及 [VERL 官方 recipe](https://github.com/verl-project/verl-recipe/blob/main/dapo/README.md)说明动态采样及生成批次上限。没有迁移其去 KL、clip-higher 或 token 计权变化。本仓库旧 `filter_group_data(last_try=True)` 会放回所有组，因此没有直接开启旧开关。

这里不改变奖励公式，也不人为奖励某种战术。显示出来的整局 reward 相等不一定表示 GiGPO 精确为零：折扣回报、回答次数、无效动作惩罚和浮点残差都可能产生非零值。候选不会将这些值自动视为零，也不声称它们都是有用的学习信号。

## 实现与用法

`verl/trainer/ppo/sc2_task_signal.py` 复用原计分函数，在复制补齐前检查真实 batch；返回检查信息，不修改传给原训练流程的 batch。accepted batch 仍按旧路径补齐、计权及计算 loss。本次不修补齐机制。判定是 pre-padding task advantage，不是 reference KL 或 PPO old-policy KL。检查保留 NumPy RNG；直接克隆张量，避免 DataProto 序列化锁住原 TensorDict。

`ray_trainer.py` 增加有限采样入口及实际更新计数。`scripts/run_record.py` 仅补充这种采样的收尾：所有消耗的游戏都计入结果，checkpoint 使用实际 step，不伪称达到 nominal target。历史入口只增加显式参数透传和 seed 记录字段校正。

通用训练入口的可选 Hydra 配置：

```bash
bash scripts/train_sc2.sh \
  trainer.sc2_skip_zero_signal=True \
  trainer.sc2_signal_max_attempts=3 \
  trainer.sc2_signal_max_batches=4
```

默认 `skip=False`、`max_batches=0` 完全保留原训练入口。正的 `max_batches` 是本 run 总采样次数上限，包含接受和跳过；每次采样是 8 局时，4 次最多 32 局。`max_attempts=3` 是单次收集机会的上限，达到后不给零组兜底、不增加 global step；下次机会仍受同一个总上限约束。空策略、NaN、SC2 错误直接失败，不走补采重试。只支持当前单组 SC2/GiGPO，禁止与旧动态采样或空策略补采叠加。

`checkpoints/signal_sampling.jsonl` 记录 collection/reset/pending-step、episode returns、有效 token/回答数及精确 advantage 最大值。`signal_sampling_summary.json` 记录采样成本、跳过数和实际更新数；已有游戏种子、原始回答及整局 outcome 日志仍保留。全零跑完预算而未更新不能通过训练验收。候选 checkpoint 可评测，暂不支持继续训练恢复采样预算；加载旧 A4 完整状态不受影响。

## 已批准的最小 A/B

两组均恢复同一 A4 完整模型、optimizer 和 per-rank 状态，使用同一个显式游戏种子表 `[1098341402, 42613, 17089, 3236598124]`，每 seed 8 局，Level 6/g2；不是从初始模型另训，也不是 bitwise SC2 resume。A/B 唯一配置差异为 `trainer.sc2_skip_zero_signal=False/True`。模型是 SpikingBrain-2.0-instruct 的 A4，非 Qwen。

其余沿用 Coverage 的 FP32 FSDP 参数、FP32 归一化、BF16 autocast、HF/eager/cache、SWA dropout=0、192-token/temperature=0.4，以及原 prompt、约15条/x N、阶段待办、开局 Planner、失败反馈关闭、history、reward、微操、学习率和 loss。两组最多各 32 局，不按接受批次数额外加游戏；A 最多更新4个 global step，B 可能更少。每个 global step 内的 optimizer update 数仍由真实回答数决定，二者不混称。

43 的 GPU 0–3 串行，合计最多3小时，最终各保存一次完整 checkpoint，预计合计约120 GiB，不删除旧产物，不自动重试或续长训。四卡匹配父 checkpoint，串行避免既有双模型主机内存冲突。`run_pair_node43.sh` 以同一截止时间调用 `run_arm_node43.sh A/B`；需设置 `SOURCE_SHA`、唯一 `SIGNAL_CONTROL_DIR` 及授权开关 `SC2_ZERO_SIGNAL_AUTHORIZED=1`。新提交只补配对编排和记录，生产候选文件与 `5130ce4` 完全相同。

部署根 `/root/new-scii-zero-signal-20260914/code-<SHA>`；控制目录 `pair-r1`，tmux `sc2_zero_signal_r1`；共享 run 根为 `/mnt/data0601/yangjinyue/new_SCII/experiments/20260914-zero-task-signal-sampling-v1/runs`。RUN_ID 为 `20260914-zero-task-signal-sampling-v1-spb2-A4-l6-{A|B}-32games-4gpu-r1`，Ray/cache 使用 `/dev/shm/zs_{A,B}_0914_r1`。模型、父checkpoint和依赖读既有本地位置，不复制大资产。20:11快照八卡空闲、共享盘约1.2TiB可用；实际启动以control及manifest为准。

先验收：零信号组确实不更新；有信号组正常连续更新；总采样不越界；全卡数值、完整 checkpoint、回答和自然终止记录完整。按全部32局报告 reward/胜率，另列被接受组，不能只报过滤后成绩。更新效率改善不等于学习改善。共同种子独立评测另报预算；若无法产生有信号批次，结论是本候选不足，不引入新过程奖励或无限加采样。
