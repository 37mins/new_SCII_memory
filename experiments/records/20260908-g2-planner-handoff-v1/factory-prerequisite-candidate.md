# Factory 前置候选：CPU 验证候选

2026-09-09用户批准停止空转，只修缺失前置导致候选清空，其余冻结。父版本为
`df35bc3`，游戏/训练 runtime 对照为 `e3ffac9`。复用当前候选分支，不改服务器
deployment，不启动新GPU实验。本变更不是已验收的RL版本。

## 唯一行为变化

`SC2ArenaDyStrategy/agents/planner.py` 的 `filter_bo_todo_for_prereqs()` 增加两行：
当观测中没有Factory，当前待办需要Factory/Starport/Viking/Tank科技链，且待办
未包含Factory时，加入 `factory: 1` 候选。沿用该函数已有的Refinery/Barracks
前置候选补充方式，不新增依赖框架，不强行修改Commander的阶段配额。

原来的高阶动作过滤保留。已有Factory时输出不变；已有Factory配额不被覆盖；
没有科技目标时不添加Factory。只看己方结构，敌方Factory不能满足前置。
prompt模板、解析器、六条/96token、阶段策略、攻击阈值、Executor、微操、reward、
数值精度、cache、优化器和loss均未改动。版本化源码清单只更新Planner文件哈希。

## 同输入 CPU 结果

使用r1首个实际空输入的观测、原待办和当前冻结阶段指令，现有fake-socket工具调用
真实Planner/validator；旧版本通过 `git show e3ffac9:<planner路径>` 在内存加载。

| 输入/结果 | e3ffac9原版 | 本候选 |
| --- | --- | --- |
| 原待办 | Tank9、Starport1、Viking8 | 相同 |
| 最终候选 | none | factory ×1 |
| 同一回答 `- factory` | 无效，执行计划[] | 有效，执行计划['factory'] |
| query变化 | 原空列表/空示例 | 仅候选与原自动示例增加Factory；逐字差异断言通过 |

另测仍有Marine候选时，合法 `- marine` 只生成Marine计划，不强插Factory。
**无效回答补齐规则保持原样，因此非法 `- tank` 仍判无效，但旧fallback现在可能
补出Factory；这是规则动作，不是模型选择。** 该边界单独有回归测试，未为了好看
提高合法率或修改奖励。

新测试先在原版复现空候选失败，再修改源码；测试自身初版的敌方文本split问题
已修正，不计为项目修复。最终执行：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests /private/tmp/sc2-air-results.HBI61m/decoder/bin/python -m unittest test_factory_prereq_candidate test_planner_air_options test_valid_bo_passthrough test_r6_planner_interface test_r6_format_reward test_g2_planner_handoff test_system_release test_opening_rl_launch -q
```

75项通过、退出0、6.831秒；包括9项新增测试、已有协议/直通/阶段/启动器和源码
哈希验证。同输入query差异另行CPU断言通过。没有加载模型或启动SC2。

## 停训与未验证部分

r1于北京时间14:28:54经用户批准中止，原启动器真实退出码1，result/.done保留
`failed`；[操作记录](opening-rl8-stop.json)注明人为中止原因，不覆盖原记录。
0次PPO更新、无checkpoint，7局自然胜、1局未完成；未完成局161次回答，其中134次
连续空候选无效。14:36:45核验八卡0MiB，残留SC2已按PID/启动时间验证后结束。
旧版本完整Git bundle已备份并验证，位置和哈希见操作记录。

这里只验证当前空候选有合法出口，没有证明Factory实际开工、坦克产出、连续RL或
胜率改善。其他科技前置、建造失败后的待办语义和真正空目标的处理不在本次范围。
下一步需另获新RUN短验证授权，保持原模型/seed/训练配置，不续用已污染的本批样本。
