# 开局 Planner：空候选循环只读审计

2026-09-09。审计对象为 `opening-rl8.md` 的 SpB Level6 r1-train8；44 GPU0–7，
冻结 runtime `e3ffac9e2da577679b9b980d66678e4cc72b06aa`，远程 HEAD/clean 已核对。
未改运行源码、配置、prompt、训练机制或进程。原始首个空输入与 CPU 结果见
[证据 JSON](opening-empty-candidates-audit.json)。

## 结论

主要原因是**阶段待办缺少科技前置，过滤后无合法动作，但调用与阶段刷新仍继续**。
这不是把提示词写得更强硬即可解决的模型格式问题。

| 环节 | 冻结代码与实证 |
| --- | --- |
| 上游待办 | `agents/strategist.py:246` 的 marine_pressure 三阶段均无 Factory/Factory Tech Lab；末阶段却含 siegetank9，开启 air options 后再加 starport1/vikingfighter8。前27次实际输入没有 Factory 候选，原始回答也未选择 Factory。 |
| 前置过滤 | `agents/planner.py:819`；当前观测为 factory0、refinery1。原待办 `{siegetank:9, starport:1, vikingfighter:8}` 经879–884行过滤为 `{}`，并未加入 Factory 候选。不是把敌方 Factory 错当己方：CPU 实测己方计数为0。 |
| 最后攻击候选消失 | 第26/27次回答均合法选择 `Launch OFFENSE with 8 marine`；`planner.py:508` 在 account_issued=1 时以 threshold×(attack_ct+1) 判定。CPU 复现军队人口17、attack_ct2时需24，所以攻击候选也为空。此处只解释触发条件，不判定应修改攻击规则。 |
| 无法刷新 | `agents/commander.py:100,111,265` 检查过滤前 waiting_bo，仍有3个键；terminal phase 禁止按时间刷新。输出BO为空，扣减循环也不改变原待办。 |
| 循环调用 | `players/llm_player.py:674` 每帧发现执行队列空就 renew_bo；`planner.py:1143` 仍将 `none` 和空例子送入模型。已修的同帧 while 不在此处复发：当前是跨帧反复阻塞推理，游戏缓慢推进。 |
| 无解判定 | `agent_system/environments/strategic_sc2_env_manager_v2.py:875` 在解析回答前直接拒绝空候选。真实回答、空回答、空 answer 模板、Factory、Marine、标准进攻六种输入在 CPU 下全部无效、执行计划均为[]。 |

表中 `agents/`、`players/` 相对冻结 runtime 的
`agent_system/environments/env_package/sc2_online_restored/SC2ArenaDyStrategy/`。

时间线：游戏07:15进入最终阶段；14:23仅剩 Marine2 与攻击候选；随后两次合法攻击；
14:32首次输入 `none`。北京时间14:11:30快照已有83次连续空候选，仍7局胜/1局未结束，
无PPO更新、无checkpoint。这里的局数不能写成训练step数，已完成7局也不能当作RL收益。

## 对训练的影响

八条rollout必须全部结束后才进入更新，因此一个环境拖住整批。空输入下的回答仍有
`policy_action_applied=true`，环境日志reward为0；`agent_system/sc2_episode.py:80`
会保留这些真实消费的回答，不按动作合法性排除。CPU 单条复现保留1条无效样本。
若这批最终进入训练，`verl/trainer/ppo/ray_trainer.py:200,1265` 按冻结配置再扣
invalid-action penalty0.1。这是系统提供无解输入后产生的惩罚样本，不是终局伪决策。
尚未发生PPO更新，不能声称权重已被这些样本训坏，也未计算其实际advantage。

## 验证与边界

本地以现有 `planner_scope`、`strict_validator`、`replay` 和 `policy_transitions`
直接调用冻结同版函数，使用首个实际空prompt/观测与日志原待办；fake socket，无模型、
GPU、游戏。命令为 `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests /private/tmp/sc2-air-results.HBI61m/decoder/bin/python -`，
stdin执行六回答对照、前置计数/过滤、末阶段刷新与策略样本保留断言：退出0。
CPU中attack_ct2由紧邻的两次合法攻击回答对应；未增加运行时插桩。

三阶段漏Factory和对应过滤在本地Git根提交df1cbb0（2026-08-19）已存在；本次开局
接管没有修改它们。根提交不是原ZIP的作者/时间证据，不据此声称最初压缩包已核验。
现有固定g2文件包含05:01/05:03/05:11三条Factory和05:45/05:53/05:55三条Factory
Tech Lab；这说明完整固定BO有提供前置的路径，而当前开局Planner阶段配置没有。
这不等于这些固定动作在每次历史游戏都成功完成。
前27次的观测、回答和带worker PID的执行日志支持“未提供Factory构建路径”，
不等于逐帧核验了全场实体事件。其他七局在10–12次决策后已获胜，未走到本次配额耗尽边界。

下一候选应只针对缺失前置的候选生成完整性，先验证 Factory 能合法出现在候选并由
模型选择，而不是 Executor 直接补造；不顺带改prompt、阶段策略、攻击阈值、奖励或训练计权。
它尚未实施或获游戏验收，不能承诺一次改动同时解决坦克产出和全部训练问题。
建议保留本RUN并停止空转，修复须另获授权、新SHA/新RUN，不覆盖失败记录。
