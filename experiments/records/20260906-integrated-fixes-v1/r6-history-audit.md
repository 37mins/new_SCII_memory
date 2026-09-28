# r6 与当前真实 RL 版本：只读审计

核验：2026-09-06 23:51 北京。未启动历史 GPU 对照，未修改模型、loss、reward、
prompt、BO 或微操；没有加入 early-stop。原始证据与逐步数字分别见
[审计 JSON](r6-history-audit.json)、[96 步 CSV](r6-training-metrics.csv)、
[原始命令与来源](r6-effective-controls.json)。

## 当前真实状态

冻结 SHA：`3e1cc0afe84c0e8b24305eaf78427d73aeec5530`。
节点 44，计划 GPU 0–7。23:30 首次命令因其他任务占 GPU 0–2 返回 1，未创建 run。
23:49 八卡释放；23:50:17 创建 `sc2_real_r2`，实际初始评测 RUN 为：

`20260906-integrated-fixes-v1-qwen-instruct2507-l6-8gpu-fp32-rl96-gate8-s0-r2-initial-eval`

**23:50:53 初始评测启动失败，退出 1；0 局、0 回答、0 PPO 更新，train8 未启动。**
不是正常任务被历史审计中断。错误为：

```text
AssertionError: real_train_batch_size (1) must be divisible by total n_gpus (8).
```

原因是我新增的八卡评测入口漏覆盖单卡评测的 `env.rollout.n=1`。
`scripts/train_sc2.sh:24` 将评测组数设成 1；入口恢复了八卡与八局 validation，
却漏掉训练器初始化检查使用的组数。CPU 单测及 Hydra 组合通过，没有覆盖
这项实际初始化检查。它是本次编排遗漏，不算项目原有 bug 的修复成果。
失败产物保留，未原位重跑；已请求仅修评测入口并用新 SHA/RUN 重启。
[实际启动与失败证据](real-rl-r2-launch.json)。

随后只读 CPU 核验调用冻结的 `RayPPOTrainer._validate_config`：train8 原配置
通过；initial-eval/step8-eval 原配置均复现断言，仅在内存补 `env.rollout.n=8`
均通过，validation/generation n 仍为 8。未加载模型、修改文件或重启任务。

## r6 成功结果已复核

指定 RUN `20260818-qwen-canonical-plan-rl-v1-qwen-base-l5-full96-s0-r6`，
原节点 198，现归档于 197 的
`/data/yangjinyue/new-scii-qwen-level5-canonical/runs/<RUN_ID>`。
runtime commit `e61a23b8c642cb1b88d67f11f1f6d9a96a65ef31`，编排 commit
`cbf7424b1ec764d2b059fbd2ffe410f98469f9c7`；实际比较的是该 run 的补丁后
`results/audit/*/frozen_runtime_source.tar.gz`，不是当前工作树冒充历史源码。
该归档 SHA-256 为 `cad52ff43ebefb0c0d7eb1cb809f2144f14c799db928467a755b8899e399ce32`，
与其伴随校验文件相符。

原始日志完整 step 1–96：**625/768 胜，81.38%，平均 reward 9.904**。
`driver.exit=0`、`.done` 的 `CHECKPOINT_COMPLETE=1`；step-96 的八卡
model/optim/extra 分片和 data.pt 共 25 项均存在且非空。

| global step | 胜数 / 128 | 训练胜率 | 平均 reward |
|---|---:|---:|---:|
| 1–16 | 94 | 73.44% | 6.219 |
| 17–32 | 102 | 79.69% | 9.082 |
| 33–48 | 102 | 79.69% | 8.532 |
| 49–64 | 110 | 85.94% | 12.275 |
| 65–80 | 102 | 79.69% | 8.486 |
| 81–96 | 115 | 89.84% | 14.830 |

存在中段回落，但首尾改善真实成立。它是单 seed、持续变化策略的训练结果，
没有独立训练前后评测，不能直接证明泛化提升或架构优势。

## 关键差异—证据—可能影响

“当前”指冻结的 train8 配置，**不是已经产出结果的训练**。源码路径均相对于
各自冻结根；r6 相关行与哈希已摘录在审计 JSON。

| 项目 | r6 → 当前 | 证据与判断 |
|---|---|---|
| 模型 | 官方 Qwen3-4B → 官方 Instruct-2507；都无项目 SFT/RL、thinking 关闭 | 两份模型 manifest。权重、官方 tokenizer/chat template 一并不同，是混杂因素；不能直接判 2507 更差。r6 并非未经官方后训练的 foundation base。 |
| 难度 | Level 5 Hard → Level 6 Harder；地图、Air 对手和 g2 不变 | r6 原始命令、当前 train8 effective config。应优先区分任务难度与代码退化，目前没有这两个完整系统的同配置难度对照。 |
| 参数／归一化精度 | r6 Actor master 已 FP32，但 FSDP 前向参数默认 BF16，温度除法在 BF16 logits 上；当前前向参数、温度归一化显式 FP32，ref 同步配置 | `fsdp_workers.py:203–207,294–305`、`dp_actor.py:221–233`。已有小批 A/B 支持局部精度改善；r32 未证明完整更新稳定，当前连续 RL 尚无结果。不是把所有运算改成 FP32，BF16 autocast 保留。 |
| HF 生成位置 | r6 显式传 prompt position_ids；当前 Qwen3 由 growing mask 生成递增位置；cache 两者开启 | `hf_rollout.py:148–154`。已有同输入生成对照 0/20→20/20 合格，支持接口修复；不等于游戏学习改善。未改 prompt 来取得该结果。 |
| 奖励 | r6 每调用按当前状态重复加最多 0.5，格式标记无条件 True；当前 per-call shaping off，格式奖励依完整回答校验 | 历史 `planner.py:get_action/generate_plan` 与当前对应函数。reward/advantage 分布实质改变，旧新 reward 不能直接比较。既有单步 A/B 验证开关生效，尚未验证长期收益。胜负奖励与终局状态 shaping 保留。 |
| 终止和策略归因 | r6 call20 直接 done 并加败局罚分，收集器只按 active_masks 收录；当前自然终止，未被游戏消费的终局回答不进入 PPO，奖励归到真实决策 | 历史 `planner.py:1109–1120`、`rollout_loop.py:333–353,474–508`；当前 `sc2_episode.py` 与 `policy_transitions` 函数。已有短局归因/奖励守恒验证；改变了训练样本与时长分布，不能假定正向曲线必然保留。 |
| 算法／计权／优化器 | GiGPO mean_norm、gamma .95、AdamW lr1e-6、betas .9/.999、wd .01、clip1、mini8/micro1、epoch1 不变；copy 补齐仍存在 | 原始命令和冻结 `core_gigpo.py`、`core_algos.py` 主体一致；`adjust_batch` 有效 AST 完全一致。没有“最近换了学习率／loss 计权导致”的证据。终止过滤会间接改变样本数，因此每 global step 的 optimizer 更新数不固定；复制补齐不能宣称已修复。 |
| Planner／Executor／history | 同一 canonical names-only、动态 one-shot、6 BO、4096/96 tokens，history0/memory off，old_micro + marine_pressure | 主 prompt 字符串相同，阶段模板只少一个行尾空格；generate_plan 唯一有效代码差异为移除无条件格式标记。解析／待办选择函数未改；ScriptPlayer 去文档字符串后的全 AST 相同，Commander 同，Strategist 逐字相同。另有独立 unit-ID 耗尽修复，非攻击波重写。没有证据表明新换了微操或新关了 history。 |
| KL 报告 | r6 reference KL 仅最后一个 micro-batch，跨卡仅 rank0；当前收集全部 micro-batch/卡 | `dp_actor.py:450`、旧 `DataProto.concat:710`。旧 PPO KL 为有效 token 上的 signed K1；reference KL 是封顶 K3，二者不是同一指标。只修汇总不改 loss，不能拿新旧 console KL 数字直接论稳定性。 |

当前 Torch/Transformers/Ray 与 r6 同为 2.7.1+cu128 / 4.55.2 / 2.48.0；
r6 Python 3.12.11 容器，当前 3.12.3 固定依赖目录，非完全相同环境镜像。
目前无证据将学习差异归因于这个运行环境差别。PPO mask/指数有界修复在 r6
冻结包里已经存在，并非当前相对 r6 新增的变量。

## r6 当时已有的异常／限制

原 console 的全程平均：PPO clipfrac 6.56%、响应长度触顶率 59.76%、
裁剪前 grad_norm 350.952（最大 2851.231，step90）；96 步梯度指标均超过 1，
随后按同一 grad_clip 裁剪。PPO K1 均值 0.142，范围 -2.553～2.518；
reference capped K3 均值 2.268、最大 6.669。上述是旧汇总口径，非完整全卡分布。
长度触顶不等于 PPO clip；这些数字没有阻止该次系统训练首尾胜率改善。

call20 人工终止、格式奖励误记、终局伪决策收录风险、copy 补齐、生成位置问题、
6 条 BO／96 tokens 和固定宏观／自动战斗依赖在 r6 都已存在。它们值得标记，
但不能据此抹掉 625 胜；反过来，625 胜也不证明这些机制正确或适用于 Level 6。
旧 BO 扣减、生产完成反馈与攻击波能力没有在本次审计中被宣称修好。

已扫描原始回答日志 **10,512 行的全部字段**及 run 的 audit/log/checkpoint
清单：没有保存逐 token 更新前后概率、对应梯度或逐更新参数差；仅有训练末
checkpoint，无法重建早期每次更新。因此 **不知道 r6 是否出现过 r32 同类漂移**，
既不能声称没有，也不能声称存在。上述 10,512 是日志行数，不是有效 PPO 样本数。

## 优先级与下一项

先解决已经确诊的八卡评测入口声明遗漏，让既定真实 RL 路线启动；不借此增加
数值机制。对学习表现优先区分 **Level 5→6 的难度差**和**奖励／终止／真实
样本分布的改变**。精度组合有局部证据但连续训练待验；“2507 天生更差”、
“新 prompt／微操改坏了”、或“r6 没有概率漂移”目前都缺证据。

step-8 若无明确退化，就按批准路线续训，不因历史审计强加额外 GPU 对照。
若输出稳定但 Level 6 表现仍差，优先提议 **只把难度换成 Level 5**：同一
Instruct-2507、当前冻结系统，4 global step ×8 rollout、每步完整 checkpoint，
初始／step4 同八个种子的独立评测；8 卡、预计 4–6 小时、最多 8 小时。
须另获预算批准，不自动排入。本对照不是重跑 r6，作用是判断当前系统是否仍在
Level 5 可用；若 Level 5 同样出现输出退化，则不再把问题简单归因于难度，
根据 step-8 的具体证据再提出唯一变量。4 步不保证足以证明学习提升。
