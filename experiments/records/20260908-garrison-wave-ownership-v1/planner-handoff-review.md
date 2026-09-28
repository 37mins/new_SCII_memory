# Planner 接管时机：g2 末阶段交接审议

2026-09-08，仅完成冻结源码、既有日志/回放和轻量CPU审议。未修改runtime、未启动
游戏或训练。控制runtime为`e1e25ce1d14362b209e34ec73871ebaeb63538a6`，分析依据见
[seed43567结果](failed-seed-analysis.md)。本方案是候选接管条件，不声称已修复学习，
也不声称恢复整个SC2_online.zip的调用链。

后续：用户已确认本方案，最小候选及 CPU 回归见
[实施记录](../20260908-g2-planner-handoff-v1/README.md)。以下保留当时审议内容，GPU预算仍未获批。

## 建议的唯一变量

| 项目 | A：现有控制 | B：建议候选 |
|---|---|---|
| 首次Planner之前 | 完整g2，237项 | 同一g2的前138项，日志编号#0–#137 |
| 交接边界 | #236按原逻辑收尾后 | #137按原逻辑收尾后、#138开始前 |
| seed43567旧轨迹中的对应时刻 | 09:52 | 06:53，实际新运行仍需验证 |
| Strategist阶段 | Siege Tank Power Spike and All-in Commitment | 相同末阶段、相同阶段文字和原始配额 |

不是游戏时钟到06:07就打断动作。g2的#137标称06:06，#138是标称06:07的第一条
Marine，#139为该段大规模进攻描述；依据是文件的原有时间/动作边界，不新增阶段
标记或切分算法。g2本身没有显式phase字段；不能把这个候选边界冒称原生调度功能。
本次16局既有日志中，#138都在06:53开始。两份存留回放在此处有3个已完成Factory、
1个已完成Factory Tech Lab、另2个Tech Lab在建；三条相关BO #123/#129/#130均已处理。
这保留了进入Tank阶段所需的开局科技路径，而不是仅清空执行队列。

不选03:30接管：当前启用的marine_pressure前两阶段是Marine rush指令，并不对应
g2宏观开局。提前到那里会让此前未执行的策略阶段生效，不能再声称是在相同末阶段
下仅改变接管边界。不选06:00硬截：在43567中三项Factory Tech Lab BO尚未处理，
末阶段又不提供这些BO；可能直接截断原有科技前置。

## 待办从哪里来

沿用现有调用链，不将g2尾部自动转换成新目标：

```text
现有Commander.renew_bo(当时真实observation)
→ 现有Strategist.generate_strategy：选择当前末阶段
→ 现有已建宏观项扣减 → waiting_bo
→ 现有Planner前置过滤、动态one-shot、query
→ 模型回答 → 原解析/规则处理 → Executor的新队列
```

末阶段原始配额来自冻结`agents/strategist.py:277`：

```text
marine 50, supplydepot 9, mule 6, siegetank 9,
SCV 15, marauder 4, commandcenter 2, refinery 1
```

本次CPU逐项核对：这恰好等于g2从#138到结尾的96条非攻击动作计数；另3条是原
进攻描述。这里只做来源一致性核验，没有编写或启用“BO转阶段目标”机制，也没有
把尾部直接喂给Executor。已启用的air选项仍按原代码另加starport1/vikingfighter8，
不是本候选新增。随后`_reconcile_completed_macro`按当时观测扣减建筑/工人，
`filter_bo_todo_for_prereqs`继续按原规则处理，Commander仍按原选择口径扣待办。

**固定剩余队列不等于模型待办。** 被交出的#138–#236保存为固定队列未执行尾部，
不标记“完成”、不补回Commander、不追加到模型的新队列。已向SC2提交的建造/训练
继续由引擎执行，不取消、不重发，也不改变“提交/完成”的现有语义。

冻结的是prompt模板、阶段文字、配额和处理规则，不是整段query逐字不变：提前
接管必然改变观测、经原规则计算的待办数量及动态例子。这些是接管时机的结果，
必须保存出来，不能冒称模型收到的输入仍完全一样。

## 最小实现边界（待确认，未实施）

只在`players/llm_player.py:64`固定BO初始化处增加默认关闭的前缀限制，候选值138。
默认保留全部237条；限制只作用于首次加载，不裁剪后续Planner队列。原g2文件不改，
原队列收尾和`renew_bo`入口不改，避免新增一套中途抢占、挂起与恢复状态机。
保存原BO哈希、保留前缀及交出尾部范围，并记录首次Planner的实际时间和输入。
候选入口只接受本次已核验g2哈希/边界，不把138静默用于g1/g3/g4/g5。

模型继续用原始SpB；prompt、6BO/96token、历史窗口、解析器、fallback、Marine guard、
20/10自动波次、驻防修复、微操、reward、cache、精度、采样参数和训练机制均不动。
本轮没有修正代码中“末阶段文字06:07、实际选择器从06:00返回末阶段”的差异；
CPU确认06:53与09:52都进入同一phase_index=2，因此不需要联动修改。

实施前备份/提交到独立候选，不改原e1e25ce deployment、历史run或g2资产。CPU回归
应证明默认仍走237项、候选只在旧动作正常收尾后交接一次、不丢当前待处理命令、不
把固定尾部当模型选择，且后续每轮沿用原Planner链。打印阶段原始配额、扣减后的
待办、query、人工指定的两份不同合法回答及各自执行队列，以验证接口，不冒充模型生成。

CPU通过后建议纯评测A/B，不做PPO：两组都用相同原始SpB、各4卡、4局；实际游戏
seed列表均为[2732,2732,43567,43567]，覆盖已有成功和失败场景，明确固定生成seed及
相同采样参数。建议八卡并行、总墙钟上限60分钟（8 GPU小时）；此预算尚未获批。
两组都重新记录，不能把此前不同局数/采样RNG的旧结果当严格A。按env核对现有日志；
既有按秒回放覆盖的限制单独注明，本候选不顺手修改日志路径机制。

验收先看交接前轨迹是否保持、交接是否确实提前、真实模型回答是否改变后续队列；
再看胜负、累计reward、单位产出/损失、有效率和典型回答。更多调用/更高有效率不是
成功定义。这个小样本只判断明显退化与机制可用性，不要求保证涨分，不推出架构优势，
也不直接批准96步。约06:53以前已发生31个Marine死亡（两份回放），本候选不会
修复那些早期损失，也不会解决自动波次或缺少升级的既有局限。

## 只读核验记录

源位置均以`git show e1e25ce:<path>`核验，前缀为
`agent_system/environments/env_package/sc2_online_restored/SC2ArenaDyStrategy/`：

- `knowledge/BO/replay_observations_Flat128_d4_TvT_Macro_g2_Result.Victory_twobase.json:830`
  是#138，SHA256 `304b1f4ff12fe30c5e6d5f7b8eaef8ed6823ca65afed3a3706cae10e3639c494`。
- `agents/strategist.py:246、273、344、350、359、405`：阶段、air备选、阶段选择和宏观扣减。
- `agents/commander.py:265、303、310、322`：待办装载、调用Planner、既有扣减。
- `agents/planner.py:819、1143`：前置过滤与原query生成。
- `players/llm_player.py:418、660、731、946`：旧BO收尾、Planner触发、攻击描述不直接
  发起波次、命令提交后推进。前缀收尾并不意味着所有单位已经完工。

CPU核验使用`git show`读取原BO/Strategist，在独立Python进程中设置原profile/air开关，
断言尾部非攻击动作计数等于现有末阶段配额；断言06:53和09:52均为phase2，全部通过。
回放沿用已有mpyq/s2protocol Base75689，在#138对应game_loop9264处截取tracker，
两份均显示31个Marine死亡、1个Tech Lab完工（06:45.18）、2个在建。原分析快照和
输入哈希沿用[上轮记录](failed-seed-analysis.md)，没有加载模型或占用服务器。
