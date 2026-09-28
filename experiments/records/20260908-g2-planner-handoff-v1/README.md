# g2 首次 Planner 接管候选

09-09：新增[参考五月动态分支的开局A/B](may-opening-ab.md)，B=-1表示不执行初始
固定BO，A=0保持完整g2。仍使用当前Strategist/prompt/系统，不称五月完整复现。
旧0/138的r3两组已退出0，产物保留；新A/B准备状态见独立段落。

2026-09-08：用户确认实施 [#138 之前交接方案](../20260908-garrison-wave-ownership-v1/planner-handoff-review.md)，本轮只做本地代码与 CPU 回归。**106 项 CPU 测试通过；未部署、未启动模型/游戏或 PPO，不是胜率或学习验收通过。**

前两次A/B尝试在游戏前失败。修正r2的采样RPC回调签名后，**r3已于09-09 10:00在44八卡启动，10:02两组均进入SC2**，共同截止10:50:26。各四局纯评测，无PPO，目前无胜负/方法效果结论。见[实际运行记录](gpu-ab.md)；上段保留初次CPU交付状态。

## 唯一行为变化

`SC2_FIXED_BO_PREFIX_LIMIT=138` 只在 `LLMPlayer.__init__` 加载固定 BO 时保留 g2 的 #0–#137。#137 仍按原逻辑提交或超时跳过，下一帧沿原队列耗尽路径调用 `Commander.renew_bo(observation)`。不按时钟抢占、不取消已经提交的命令；后续 Planner 队列不再切片。

默认值 `0` 保留完整 237 项。候选只接受 `138` 和 SHA256 为 `304b1f4ff12fe30c5e6d5f7b8eaef8ed6823ca65afed3a3706cae10e3639c494` 的原 g2；其他非零边界、非整数值或其他 BO 内容明确报错。原 BO 文件不改。初始化日志保存原哈希、保留数量和未执行尾部的半开区间 `[138, 237)`；尾部可由原资产恢复，不记成完成，不转换成新待办。

待办仍来自原 `marine_pressure` Strategist，经原宏观扣减、前置过滤后进入原 Planner query。06:53 的 CPU 输入选择原末阶段 `Siege Tank Power Spike and All-in Commitment`。原末阶段的 96 条非攻击配额与 g2 尾部对应，已有 air 选项另提供 starport1/vikingfighter8；本改动不生成配额。原始阶段、处理后待办、完整 query、两份回答和返回队列见 [CPU 接口证据](cpu-interface-evidence.json)。其中 observation 是人工测试输入，回答也是手写，不冒充历史完整观测或实际模型生成。

同一个 query 的两份合法回答分别得到：

| 人工回答 | 原 Commander 返回的 Executor 队列 | 选择计数 |
|---|---|---|
| 六行 `- marine` | 六个 Marine，未补 Tank/Viking | marine6 |
| `- marine`、`- starport` | Marine、Starport，保持顺序，不补满六条 | marine1、starport1 |

没有同时修改 prompt、6BO/96token、策略文字、待办扣减、解析器、有效回答直通/无效补齐、Marine guard、20/10 自动波次、微操、reward、模型、cache、精度或优化器。新游戏的 observation、过滤后的待办和动态例子会随交接提前而变化，不能宣称最终 query 与旧游戏相同。

## 代码、使用与回滚

候选分支 `codex/experiment-g2-planner-handoff`，工作树 `/private/tmp/sc2-planner-handoff-20260908`；父提交 `d2e660dfb17ef13c7193245a52450e6628562a93`，保留在原 `codex/experiment-planner-air-options` 分支。游戏层基于已审议的 `e1e25ce1d14362b209e34ec73871ebaeb63538a6`：父提交中的游戏层与其相同，但仓库还含此前默认关闭的训练空批次候选和记录，**不能把整个父提交称为逐字相同的 e1e25ce runtime**。本轮未改这些训练模块；未来纯评测 A/B 应使用同一个候选 SHA 的开关 0/138，避免混入源码版本差异。

生产代码仅两处：`players/llm_player.py` 初始队列限制，`scripts/train_sc2.sh` 导出参数并传入 Ray `runtime_env.env_vars`。源码哈希登记只更新 `configs/system-v1.sha256` 的 llm_player 项，新哈希 `4943119f134e088d600bac3d056c7f49eb731ffeb467f4f861d8e82fc3822285`。

在未来获批的冻结候选入口前设置 `SC2_FIXED_BO_PREFIX_LIMIT=138` 即可启用；`0` 或不设置则关闭。Qwen/SpB 共用入口均已 dry-run 验证，effective config 会记录 `+ray_init.runtime_env.env_vars.SC2_FIXED_BO_PREFIX_LIMIT`，A/B 去除输出目录差异后只有该参数不同。旧服务器冻结入口仍指向 e1e25ce，**本轮没有部署，不能直接在旧入口上设置参数就宣称候选运行了**。

实施前完整 Git 备份 `/private/tmp/sc2-handoff-baseline-d2e660d.bundle`，`git bundle verify` 通过；SHA256 `eb514958c239f8ecec0d8f25c718bb728d77f3382288b246c6fa9cb5e62e9bf8`。原分支、deployment、run 与失败记录未动。CPU 候选通过后本地原子提交；本轮没有 GitHub 推送或服务器写入。回滚使用父提交或旧冻结 runtime 对应的配置，不手工覆盖旧目录。

## 已执行验证

在候选工作树执行以下命令，`PYTHON` 为现有轻量环境 `/private/tmp/sc2-air-results.HBI61m/decoder/bin/python`，均退出 0：

```bash
PYTHONPATH=.:tests "$PYTHON" -m unittest test_g2_planner_handoff test_empty_bo_frame test_valid_bo_passthrough test_system_release.LauncherTest -v
PYTHONPATH=.:tests "$PYTHON" -m unittest test_r6_planner_interface test_planner_air_options test_air_plan_attribution test_r6_format_reward test_r6_natural_termination test_sc2_evaluation_without_policy test_garrison_wave_ownership test_barracks_production_slots test_pending_addon_placement -v
PYTHONPATH=.:tests "$PYTHON" tests/test_g2_planner_handoff.py --evidence
```

前两组分别 40/40、66/66；提交前合并执行同一组模块为 106/106，完整输出保存于 [cpu-tests.log](cpu-tests.log)。`bash -n scripts/train_sc2.sh`、全部 21 项 `shasum -a 256 -c configs/system-v1.sha256` 及 `git diff --check` 亦通过。新增回归执行真实构造器/回调和 Commander/Strategist/Planner/validator 方法，模型 socket、SC2 对象/命令边界用 CPU double 替代。验证默认 237、候选精确保留 138、当前 BO 等待时不交接、沿原超时语义收尾、每帧最多一次刷新、后续队列不重切、固定尾部不计入选择、不同合法回答改变执行队列。故意构造的 140 条后续队列仅用于捕捉重复切片，不表示放宽六条输出协议。

SC2 引擎实际接受、单位产出、首次接管时间及 Ray 多进程实测仍未验证。#138 在已有 seed43567 对应 06:53，不保证其他 seed 同时到达；阶段仍由原时间选择器决定，不新增强制 phase2。新局必须检查实际阶段及科技状态，若提前落入另一阶段，先报告，不顺手改策略。

## 下一项，已获资源批准

原始 SpB、Level6/g2 的纯评测 A/B；同一候选 SHA，A=0、B=138，其余复用历史冻结数值/生成/系统配置。各四卡四局，同一游戏 seed 列表 `[2732,2732,43567,43567]`，相同生成种子与采样参数；八卡并行，总墙钟上限 60 分钟（最多 8 GPU 小时）。用户已批准这一预算；[运行记录](gpu-ab.md)区分准备与实际启动，**不启动 PPO 或 96 步训练**。

先比较交接前轨迹、实际阶段、首次 Planner 时机及回答到队列的作用，再比较胜负、整局 reward、产出/损失和回答退化。更早/更多调用或更高有效率本身不是成功。该小样本不保证涨分，不能证明架构优势，也不修复交接前已经发生的送兵。
