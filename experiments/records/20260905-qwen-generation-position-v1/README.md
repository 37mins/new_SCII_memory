# Qwen 缓存生成位置修复

2026-09-05 23:30 北京时间，running。用户批准修复并做真实训练短测。基于 A/B 实际运行的
`c24bec5183048bf8b409abde88bb18247e137cb0`，隔离分支
`codex/fix-qwen-generation-position`，不带入 KL 统计候选。涉及生成策略语义，
故与旧只读运行时隔离；不是为了增加一次运行而新建分支。

唯一生产改动：`HFRollout` 对 `model_type=qwen3` 不再向 `generate()` 传固定
prompt position_ids，让 Transformers 根据不断增长的 attention mask 构造位置。
cache 仍开启。返回给后续 log-prob/PPO 的位置张量、模型加载、其他架构分支、
prompt、parser、6 条 BO、96 tokens、reward、Executor、微操和优化器均不变。
自然终止、终局伪决策过滤、格式奖励及完整 query 日志继承 A 的已验证修复。
现有 batch-copy 凑整仍保留，不将本次称为完整修好的训练器。

依据是 [Transformers v4.55.2 的官方生成实现](https://github.com/huggingface/transformers/blob/v4.55.2/src/transformers/generation/utils.py)
与已归档的 [四卡对照](https://github.com/CeraveNew/new_SCII/blob/0c2e52be02e56eb88a17041a51c1a9a121a8da78/experiments/records/20260905-r6-format-reward-v1/generation-probe.md)：
同十个真实输入、两个种子，原路径 0/20 合格、15/20 到达预算；自动位置
20/20 合格、0/20 到达预算。原路径 decode 位置反复为 980，对照递增。
这是独立生成证据，还不是 FSDP、SC2 或 PPO 通过。

## 本次短测

44 的 GPU 4–7，原始 Qwen3-4B-Instruct-2507 cdbee75，无项目 SFT/RL，
Level 5 Hard/Air、Flat128、canonical g2、seed 0、8 条 rollout、1 个全局训练
step。仍用 HF/eager、no-thinking、lr=1e-6、grad_clip=1、GiGPO mean_norm、
legacy step shaping；不从 A/B checkpoint 续训。与 A 的非路径 effective
config 必须相同，唯一行为差异来自本次 HFRollout 一处条件修改。

RUN_ID=`20260905-qwen-generation-position-v1-qwen-instruct2507-l5-legacy-4gpu-rl1-s0-r1`。
部署 `/root/new-scii-qwen-generation-position-20260905/code-<SHA>`，输出位于
`/mnt/data0601/yangjinyue/new_SCII/experiments/20260905-qwen-generation-position-v1/runs/<RUN_ID>`。
入口 `run_node44.sh`，复用既有 legacy arm，不复制启动参数。预计 25–50 分钟
（含 checkpoint 保存），现有两小时超时不变。不得中途修改运行时或复用目录。

验收：真实消费的完整回答按原 validator 复核，分别报告全局自然胜率和
Planner 参与局数；8 局自然结束、真实归因一致、有限更新、退出码 0、完整
checkpoint。短测不证明长期胜率提升。若没有任何真实 Planner 决策则不算
模型接口验证通过。主指标为真实完整合格率及预算截断率，而非先追求 8/8 胜。
独立 CPU 回归验证实际 HFRollout 的递增位置，并比较 cached/full-forward logits；
不能将小模型 CPU 等价冒充大模型分布式 log-prob 精度验证。

备份 `/private/tmp/new-scii-before-qwen-position-0c2e52b-20260905.bundle`，
SHA-256 `d18a0ae302cfd0304ca20fd57f67535259ee75fdc4246821f3772ea36f3ff7d5`，
包含对照 c24bec5 和此前诊断记录，依赖 base `46ce2cc`。旧分支、结果和模型
全部保留。最初授权仅覆盖短测；随后用户明确批准
[夜间 8-step → Level 6/96-step 条件接续](overnight-plan.md)，仍不自动合入其他修复。

## 验证记录

本地执行 `python3 -B -m unittest discover -s tests -v`：55 项通过、
2 项因 Torch/Transformers 不在本机而跳过，旧 `test_release_contract`
因缺少 gymnasium 无法导入。不能据此称全套通过；部署后使用服务器固定依赖
重跑。新 launcher 的 `bash -n`、`git diff --check` 通过。

服务器首次全套回归 60 项通过、1 项在新增 CPU 测试结束清理时失败：
`torch.cpu` 没有 `empty_cache`。只修测试支架，屏蔽 GPU cache 清理，
不改变生产代码或模型运算。该次日志保留为部署目录 `cpu-tests.log`；正式
短测尚未启动，将从新的已提交 SHA 重跑回归。

最终部署 `code-c3315e8`，已提交并推送 SHA
`c3315e8670e0afb8d58af75bf54e7756069854f5`。同一命令在固定 Linux 依赖下
61/61 通过，无跳过；CPU 小 Qwen 的 cached/full-forward logits 在
`atol=rtol=1e-5` 内一致。全套资产哈希校验通过。128 项 override 排除路径后
与 A 全同，模型、版本、BO、GPU、预算和记录中的核心源码哈希一致；HF rollout
的新哈希另由 `configs/system-v1.sha256` 核验。

真实运行于 23:30:03 启动，tmux `qwen_pos4_0905`；`formal=true`、dirty=false，
实际配置与预检一致。启动证据及哈希见 [launch-r1.json](launch-r1.json)。
完整回答、对局和 PPO 结果尚待结束后审计，不能提前称验证通过。
