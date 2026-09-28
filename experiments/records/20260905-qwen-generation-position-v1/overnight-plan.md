# 2026-09-05 夜间目标

用户在当前位置修复短测启动后，明确批准持续监管至北京时间 09-06 08:00，
通过后接 Level 5 的 8-step，再通过后启动 Level 6 的 96-step。资源仅为
sc2-a100-44 GPU 4–7。复用本 record 和分支；每阶段独立 RUN_ID，原运行不改。

| 阶段 | 入口 | 与前阶段的训练配置差异 |
|---|---|---|
| 当前 Level 5/1-step | `run_node44.sh`，冻结 c3315e8 | 仅 Qwen 生成位置修复 |
| Level 5/8-step | `run_stage.sh l5-8` | total_epochs/save_freq 变为 8 |
| Level 6/96-step | `run_stage.sh l6-96` | Hard → Harder，total_epochs/save_freq 变为 96 |

两个新阶段均重新加载共享盘原始 Instruct-2507 cdbee75，`resume_mode=disable`，
不是持续训练；仍为每 step 8 条 rollout、seed 0、canonical g2、HF/eager、
cache 开启、4096/96 tokens、6 BO、legacy reward、原微操和优化器。
只新增薄启动器，不修改旧 run_arm（它强制 Hard/1-step/2h）。新阶段的运维
超时分别为 6h/48h，仅防任务无限挂起，不是修改自然游戏终止条件。checkpoint
只在阶段末保存，避免把 1-step 的每步保存开销照搬到长训；退出审计仍检查
完整的最终四卡 model/optim/extra_state 和 data.pt。

晋级前审计 .done、result、全部 step 指标、checkpoint、真实自然终局和
消费 ID/奖励守恒；须存在真实 Planner 决策，且完整回答问题确实改善。
逐 step 复算合格率与截断率，审阅 KL、PPO clip、梯度及原始输出是否退化。
不以胜率单独放行，不要求每步胜率上升，也不把有限梯度超过裁剪阈值直接
判为故障。证据不足或出现新的阻塞性错误时不启动下一阶段，保存证据并询问。
1-step 与 8-step 都通过，才进入 Level 6；不能只凭进程退出码晋级。

先以 DRY_RUN=1/VERIFY_ASSETS=0 产生独立预检配置，逐项与上一阶段比较。
实际启动前再次核验卡空闲、无同阶段 run/tmux、clean/pushed SHA；启动后
核验真实 manifest/config，不重复启动。运行目录沿用 README 的父目录，
默认 RUN_ID 只将 l5/rl1 替换为 l5/rl8 或 l6/rl96，tmux 分别为
`qwen_pos_l5_8_0905`、`qwen_pos_l6_96_0905`。

已准备的 KL 统计候选仍隔离在旧开发分支，出处
`0c2e52b:experiments/records/20260905-r6-format-reward-v1/kl-metric-preparation.md`；
它的 CPU 测试不替代当前分布式验证。不得直接夹带进本次扩量，亦不为准备
该项阻塞已通过的运行衔接。当前 actor/kl_loss 仍是旧末 micro-batch 口径。

当前任务的目标已建立；heartbeat `sc2-r2` 已更新为夜间监管，每 5 分钟
检查一次，重要变化才通知。到 08:00 交接并暂停 heartbeat，不停止已经启动
的 96-step；若届时前置门槛未通过，报告未完成原因，不自动越级。后台监管
依赖本机及 Codex 应用保持运行；服务器内的 tmux 训练独立于客户端会话。
