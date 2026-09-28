# 下一项：Planner 输入与动作协议核对

2026-09-05，r2 计划状态。r1 已完成，用户随后要求接下一项。沿用同一实验
记录和分支，应用已准备的日志单行改动；不修改 r1 冻结运行时或结果。

## 已核实

`planner.py` 的完整 query 已包含可用单位表、当前待办、从当前待办生成的
one-shot，以及逐字复制名称、禁止别名和 `x N` 的要求。不能再声称这些内容
在当前源码中缺失。无 memory 的 manager 原样传递 query；rollout 将它放入
user message 后调用 tokenizer 的 chat template。

新测试执行真实 Planner、validator、parser 和 manager 方法，用假 socket/观测
隔离模型与游戏。标准名 `marine/supplydepot/siegetank` 同时被 validator 和
parser 接受；`tank/depot/marine x 2` 同时被拒绝。合法前缀加非法后缀时，
整份回答不合格，但 parser 可以保留合法前缀；后续规则也能补出执行队列。

例如同一合成观测和待办下：

```text
回答 - tank
→ validator: false
→ parser: []
→ 规则后计划: [supplydepot, marine, siegetank, supplydepot, marine, marine]
```

这不是现场回放，也不证明规则队列已在 SC2 中执行。它解释了为什么
“有 BO 被执行”不等于“模型回答有效”；现阶段没有证据支持放宽 validator。

## 下一处最小准备

14:10:37 UTC 读取当前 run：56 条真实回答、18 条完整有效。56 条 prompt 日志
长度均为 2999–3000 字符；只有 16 条保留了待办标题，没有一条保留 `<answer>`
示例或最后的 `Additional constraint:`。这些是**日志截短**，不是模型输入
截短；不能拿日志缺失反推模型没看到示例。所有回答来自首次更新前的 rollout，
有效率变化不证明学习收益。

[原候选补丁](full-query-log-proposed.patch) 仅把日志字符上限从 3000 改为 0
（现有语义为不截短）。现已应用到开发分支；它不会调整 prompt、生成预算、parser、
reward、微操或 PPO。离线测试确认日志裁剪不改变 manager 保存/转发的 query。
先取得完整输入证据，再决定是否需要真正的 prompt 单变量改动。历史完整
query 无法从截短记录可靠恢复。不能宣称此项日志修复会提高动作有效率或胜率。

## r2 单次短测

RUN_ID=`20260905-r6-format-reward-v1-qwen-instruct2507-l5-legacy-fullquery-4gpu-rl1-s0-r2`。
只用 44 节点 GPU 4–7，Level 5 Hard/Air、Flat128、canonical g2、seed 0，
8 条 rollout、1 次 GiGPO/PPO 更新。权重仍是原始 Qwen3-4B-Instruct-2507
`cdbee75`，不从 r1 checkpoint 续训。资产、依赖及配置继承
[r1 manifest](launch-20260905.json)，唯一非路径配置差异必须是
`+env.prompt_response_log_max_prompt_chars: 3000 → 0`。

启动入口保持 `run_node44.sh`，显式传入上述 RUN_ID 和
`RAY_TMPDIR=/dev/shm/fmt0905_fullq_r2`。新 deployment 位于
`/root/new-scii-r6-fullquery-20260905/code-<SHA前缀>`，run 位于原实验根下
`runs/<RUN_ID>`。启动前提交并推送、生成 dry-run 配置并比对；实际 manifest
再次核查 clean SHA、模型和预算。预计 30–60 分钟，现有两小时超时不变。

验收看完整真实 query 是否保留阶段待办、one-shot、最后的数量约束，现有
validator 对完整 query/原回答的复算是否与现场日志一致；同时复核自然终止、
消费归因、有限更新和完整 checkpoint。任何错误均保留 run 并停止接续。
不自动调 prompt，不提升步数，不启动第三次实验。reference KL 的升高另行
核查，不能因日志变完整就称 PPO 已健康。

改动前备份 `/private/tmp/new-scii-pre-fullquery-6f2e880-20260905.bundle`，
SHA-256 `b0995b660f10448d9d0b9fe13687c7c03af63c20df17794ff15af9f6a0053a96`，
需要仓库已有 base `c5c4037978a8a97ef788d38f905772736ccb1900`。r1 源码 SHA
`46ce2cc`、历史结果和 checkpoint 均保留。

验证命令：

```bash
python3 -B -m unittest discover -s tests -p 'test_r6_planner_interface.py' -v
python3 -B -m unittest discover -s tests -p 'test_r6*.py' -v
python3 -B -m unittest discover -s tests -p 'test_system_release.py' -v
```

准备阶段新增 4 项全部通过；本地合计 52 项通过、1 项 Torch/NumPy 集成测试明确跳过。
此前冻结运行时的 53 项 Linux 测试不冒充本次新增测试的 Linux 结果。
应用日志补丁后重新执行上述命令：本地 53 项通过、1 项 Torch/NumPy 测试
跳过；shell 语法和 `git diff --check` 均通过。生产目录的 diff 只有
`scripts/train_sc2.sh` 中日志上限这一行，其余被冻结文件的哈希测试通过。

## 自动跟进

用户要求关闭后，应用自动任务 `sc2-r2` 已设为 `PAUSED` 并回读确认。
它原先每两分钟把检查指令发送到当前任务；不是训练器发送的内容。
服务器实验独立运行；本次 r2 由用户本轮请求启动，不恢复自动任务。
