# r4：Planner 决策为何从 306 降到 96

2026-09-07，只读核验既有配对评测与冻结源码，无 GPU、游戏或运行代码修改。
重点对齐 seed `2621640384`（env3）的 58→3 次实际消费决策。

**这对局的直接原因不是 Planner 同帧卡住：step8 模型第一次回答前，战局就已严重
受损；随后执行器等待资源/执行者/设施，游戏更早自然失败，规划机会随之减少。**
不能将全八局的调用下降全部归为这一原因，也不能据此认定模型没有退化。

## 首次模型参与之前已经分岔

| 目标 seed 的事实 | 初始模型 | step8 |
| --- | --- | --- |
| 首次 Planner 输入的游戏时间 | 10:00 | 10:39 |
| 此前直接处理的固定 g2 BO | 237 项 | 237 项 |
| 首次输入中的工人 | 48 SCV | 2 SCV |
| 首次输入中的战斗单位 | 20 Marine、2 Marauder、2 Tank | 1 Tank |
| 首次调用前自动攻击触发 | 8 次 | 3 次 |
| 自然失败时间 | 22:23 | 12:45 |
| 实际消费 Planner 决策 | 58 | 3 |

“处理 237 项”指执行队列走过索引 0–236，包含完成/跳过，不代表 237 项都建成。
初评 worker 第 13212 行为末项 `Completed/Skipped`，13221 行才首次
`Generate new plan... [0]`；step8 对应为 13415、13419 行。上述兵力来自
这一轮**模型回答之前**的最终 query，不是模型采取动作之后的观测。

两局第一处固定 BO **开始时间**差异已出现在 #10 SCV：01:32 对 01:36，远早于
任何 Planner 回答。因此，即使八个 SC2 seed 完全配对，也不是模型在完全相同的
接管状态下作答。初评 1 局、step8 2 局整局都未消费 Planner 回答；这些局仍计入胜负。

## step8 的三次调用都接到了执行器

第一次（10:39）和第二次（10:52）的原始回答完全相同：

```text
- marine
- supplydepot
- mule
- siegetank
- marauder
```

冻结规则处理后的实际队列是：

```text
supplydepot → marine → marauder → marine → supplydepot → siegetank
```

| 调用 | 实际执行进展 | 下一次调用/结束 |
| --- | --- | --- |
| 10:39 | 第一项供给站遇到 `Insufficient Minerals (50/100)`；随后六项完成/跳过 | 10:52 再次调用 |
| 10:52 | 供给站反复 `Agent returned no executor`；随后 Tank 缺设施，11 次尝试后超时跳过 | 12:01 再次调用 |
| 12:01 | 新队列为供给站、兵营、Marine、科技挂件、Marauder、Marine；兵营无执行者等待，12:25 到 Marine 时仍缺设施 | 12:45 自然失败，队列尚未结束 |

三轮共进入 15 项，14 项被标记完成/跳过；最后一项等待时终局。游戏帧持续推进，
没有看到同帧空 BO 循环或队列结束后漏掉下一次请求。`no executor` 不等同于
已经证明“完全没有工人”；原日志能证明没有选出执行者，未必解释其唯一原因。

“五行原文→六项队列”来自现有补全和重排，不能把实际队列逐项都算成模型原文。
这是冻结行为，不是本次审计新增的问题或修改。

## 奖励下降不能直接等同战斗力下降

全八局胜率仍是 **0/8→0/8**；整局平均 reward 为 **−47.107→−49.619**。
八局累计 reward 为 −376.859→−396.950，下降 20.091，其中：

- 非终局 reward 合计：**19.0→0**，占总下降约 94.6%。
- 终局当次 reward 合计：−395.859→−396.950，下降 1.091。

目标 seed 的非终局 +3.5 对应七次 +0.5，日志同时出现七项攻击 BO；step8 为零。
冻结 Commander 对解析出的攻击指令标记 `has_attack` 并给 +0.5，而 Executor 的
ATTACK 分支直接推进队列，实际出兵时机交给自动战斗。因此这项奖励不是击杀或
有效进攻收益。没有把全八局的 19.0 都未经逐项归因便宣称为攻击奖励。

本版逐调用状态 shaping 已关闭，不能把这 19.0 再说成旧状态分数重复累加。
终局当次也含格式/状态项，不是纯战斗分数。格式合格率 93.46%→100% 同样不构成
学习成功；不同接管状态下的回答变化，也不足以单独证明学习退化。

## 冻结代码证据

SHA：`ca7885d886aaaafb39fc105b1d5d46cddf4bcc5a`。
部署：44 的 `/root/new-scii-empty-bo-frame-20260907/code-ca7885d`。
下列路径相对于其 `agent_system/environments/env_package/sc2_online_restored/`：

| 位置 | 已确认行为 |
| --- | --- |
| `SC2ArenaDyStrategy/players/llm_player.py:63–64,660–664` | 开局加载固定 BO；队列耗尽才调用 Commander/Planner |
| `SC2ArenaDyStrategy/agents/planner.py:739–797` | `constrain_bo_to_todo` 补全、限制并调用 `build_scheduled_sequence` 重排 |
| `SC2ArenaDyStrategy/players/llm_player.py:418–423,731–769` | 完成/跳过推进；ATTACK 不直接发起进攻；等待超时也推进 |
| `SC2ArenaDyStrategy/agents/commander.py:280–283,295–308` | 攻击指令标记及 +0.5 奖励，不要求实际战斗成功 |
| `SC2ArenaDyStrategy/agents/planner.py:1048–1065,1110–1126` | 终局奖励构成、关闭逐调用状态奖励、接收模型回答 |
| `envSC2.py:164–171,536–546,675–677` | SC2 seed 已传入；所查子进程入口未按局设置 Python RNG |
| `SC2ArenaDyStrategy/players/script_player.py:2880–2898`；`players/llm_player.py:443–451` | 侦察和选建造基地使用 `.random`；是接管前分岔的可疑来源，尚未证明因果 |

固定 g2 文件有 237 项，SHA256 为
`304b1f4ff12fe30c5e6d5f7b8eaef8ed6823ca65afed3a3706cae10e3639c494`。
不是将阶段待办送入模型后再执行，也不是“开局就由模型生成下一段 BO”的调用链。
本次不改变该链，不因发现其局限就直接清空队列或替换阶段策略。

## 结论、限制和唯一下一项

已确认：模型介入很晚；重点对局在介入前状态就严重不同；三次回答均进入执行链；
奖励下降主要落在非终局项。未确认：接管前差异的唯一来源、其余七局调用变化的
逐项原因，以及训练对相同输入的战略选择究竟变好还是变坏。这不解释 KL 最初漂移。

下一项建议先只读对齐 **首次 Planner 调用前的固定 BO 执行差异与随机性入口**，
解释为什么同一 seed 交给模型的不是同一局面；已有 worker 日志可继续利用，
GPU 预算 0。没有具体实现偏差前，不改 PPO，也不把提前接管、改战术或重跑评测
夹入本项。若要改 Planner 接管时间，须另行确认阶段策略来源和单变量方案。

后续[固定BO只读定位](r4-fixed-bo-divergence.md)将首次聚合状态差异定位至00:23，
并核实首处兵营延迟是155矿对140矿导致；Python子进程RNG未按局绑定。部分BO
时间随后重新对齐，不能把初次延迟直接当成败局根因。step8目标局详细trace被
其他局覆盖，真实首次建造落点仍缺失；没有实施RNG候选或启动新局。

## 可复查产物

[逐调用 JSON](r4-planner-participation.json) 包含全部 16 局的首次输入摘要、目标局
58/3 次原始回答和队列、首末完整 query、worker 行号与 SHA256、冻结源文件哈希、
两次评测 manifest/config/seed/outcome 哈希。使用未去重的独立 Ray worker 输出，
逐条与 env3 的 `prompt_response.jsonl` 回答匹配，避免控制台跨 worker 去重。
八局共享的秒级 SC2 日志/回放目录有覆盖风险，没有把其中单份回放当成此 seed 的证据。

实际执行（44，CPU 标准库，不加载模型；退出 0）：

```bash
ssh sc2-a100-44 'CUDA_VISIBLE_DEVICES="" timeout 30s /usr/bin/python3 -B -' \
  < experiments/records/20260906-integrated-fixes-v1/audit_r4_planner_participation.py
```

断言通过：两次 eval 完成/退出 0、manifest 冻结身份和源文件/BO 哈希、237 项队列
顺序、全部真实消费计数、目标 seed 全 58/3 条回答对应、首次模型输入时间匹配。
本地脚本 AST 解析和 `git diff --check` 通过。这是日志审计验证，不是新增游戏验收。
