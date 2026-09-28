# 原始SpB对A4：补齐同一失败输入

2026-09-12，completed。用户批准43 GPU0–3、最多15分钟，不自动重试。
RUN_ID=`20260911-execution-failure-feedback-v1-spb2-original-for-A4-fixed1x8-4gpu-r1`。

上一轮[示例顺序对照](example-order.md)的A组已具备A4权重、**原示例**、05:49
人口阻塞输入的8条回答。本轮只生成原始SpB的8答，与这8条既有A4逐seed配对；
不重跑A4，不采用上轮被否决的B示例。唯一模型差异是原始权重对4-step RL后权重。

## 冻结条件与证据

原始SpB为已校验的`Panyuqi/SpikingBrain-2.0-instruct`，不是项目SFT或RL权重；
复用源run已有模型目录与manifest `40d6f1417568d5fda1107cdafd66c7441a679b03fb148851c8f226ecc9cf884c`。
A4为`spb2-original-l6-g2-A-rl4-4gpu-s0-r2/checkpoints/global_step_4`，父训练
runtime `361cb7ae7fd86fe8811d6fcb6946d0241bf93e11`。既有A4输出runtime为
`0ac4c73afacf94914a1ddfa1015b152cc764e5fe`、4rank均完成加载，Adam计数15不变。

输入来自firstattack40、env0、step_index2、05:49；源日志和query哈希沿用
`example_order_probe.py`。直接复用原query，删去本诊断case里未使用的B候选字段，
不改变待办、反馈、原示例或正文。原始/A4的采样seed均12000–12007，每rank2答，
不是8个独立游戏；同一游戏seed42613只是历史场景来源，不启动SC2。

HF/FSDP、FP32参数/温度归一化、BF16 autocast、cache、SWA dropout0、T0.4/
top_p0.9/top_k0、192-token、history0/off与A4已完成评测相同。新runner在生成前
逐seed核对token哈希和CUDA RNG与旧A组一致。仅初始化原始模型、不调用checkpoint
loader或optimizer；Adam state应始终为空。不更换采样参数、不增加预算或游戏。

## 实施和判断

复用`fixed_feedback_compare.py --paired-a4-run <已完成run>`，仅增加单臂生成和
既有A4配对读取。`original_a4_probe.py`校验旧产物哈希、.done、模型恢复记录并
聚合结果；不重新搭建模型、解析或游戏框架。正式Planner/Executor/reward/训练
及旧部署不改。启动复用`run_example_order_node43.sh`，设置
`ORIGINAL_A4_PROBE=1 ORIGINAL_A4_AUTHORIZED=1 SOURCE_SHA=<new clean SHA>`。
新源码独立部署并先备份，输出唯一目录，15分钟硬停止。

主要比较**完整（EOS且闭合标签）、合法、Depot先于耗人口动作**，同时保留现有
validator接受率、首动作、展开数量、重复和截断。这不是胜率或实际建成指标。
原始更好只支持本输入的训练后可靠性下降，不自动证明PPO实现bug；相同或A4更好
则不支持本输入退化假设，也不证明泛化/游戏学习已改善。报告逐seed改善/退化，
只有一个诊断场景，不用其少量抽样宣称整体能力或架构优势。

上轮A4原示例：接受8/8、完整8/8、Depot优先5/8、完整合法纠错5/8。
本轮不改代码机制、不启动长训；后续方向依据配对结果决定。

CPU：本地5项评分/示例/配对回归通过；旧A4全部登记产物SHA、八条原示例输出、
加载和计数记录通过。唯一query SHA与原件一致，未使用B示例；shell语法和diff检查
通过。正式配置还需43预检与实际原始模型加载，CPU通过不等于已生成。

## 实际启动

runtime=`ba736098909e9cf06c864ccf3dfec91d763e73d7`，43独立部署
`/root/new-scii-execution-feedback-20260911/code-ba73609`；旧deployment未修改。
备份`/root/new-scii-execution-feedback-20260911/original-a4-runtime-r1.bundle`，
SHA256=`fcf082dd76930a3c2a6679dae2c4563f630a332c0d90db3e445046256d77ae12`。
服务器5项CPU回归、全部System-v1源码哈希检查通过；与旧A4的整个解析后配置相同。
1478token及其哈希与旧A4一致；11个模型分片均指向既有原始SpB资产，不是RL分片。

18:36/18:41北京时间43无compute进程/tmux，八卡0MiB；可用主机RAM448GiB，
系统盘余94GiB。启动前再次确认GPU0–3空闲及新RUN未使用。
18:44:28启动`sc2_original_a4_r1`，硬截止18:59:28；18:44:59四rank实际开始生成。
四rank均为原始模型、未调用checkpoint加载、Adam状态空，dropout0/FP32/cache保留。
原始产物位于43本地`/root/new-scii-execution-feedback-local-20260911/shared/yangjinyue/
new_SCII/experiments/20260911-execution-failure-feedback-v1/runs/<RUN_ID>`。
18:47:35完成、退出0，耗时3分7秒，18:49:14复核八卡0MiB、无compute进程/tmux。
没有重复生成A4，新增8答；旧8答的原始日志不修改。

解释边界：A4训练时失败反馈关闭，此次与原始模型共同评测含失败反馈的同一输入。
即使观察到差异，也不能据此声称A4专门学过并遗忘了这个反馈场景；训练分布与
本诊断输入的覆盖关系需要从既有训练记录核对。

## 结果：本输入不支持A4纠错退化

| 同一输入/原示例、相同8个采样seed | 原始SpB（新补） | A4（复用） |
| --- | ---: | ---: |
| 现有validator接受 | 7/8 | 8/8 |
| 自然EOS且有结束标签 | 7/8 | 8/8 |
| 合法Depot优先 | 3/8 | 5/8 |
| 完整、合法、Depot先于Marine/SCV | 3/8 | 5/8 |
| 达192token截断 | 1/8 | 0/8 |

A4相对原始：seed12002/12004/12007改善，12006退化，另外4对主指标不变。
这是条件抽样计数，不是游戏胜率，也不是跨场景的总体提升或统计显著性结论。
原始权重并未解决所有顺序/重复问题，因此不能把这些现象全部归为项目RL造成。

真实输出（保留数量，省略标签/标题）：

```text
seed12002 原始: Marine1 → Depot1 → Barracks1 → Reactor1 → SCV1
          A4:   Depot1 → Barracks1 → Reactor1 → SCV1

seed12006 原始: Depot2 → Barracks2 → Reactor2 → SCV3 → Marine3
          A4:   Marine2 → Depot2 → Barracks2 → Reactor2
```

原始seed12004把`Depot1 → Barracks1 → Reactor1 → SCV1 → Marine3`重复多轮，
192token仍无EOS；Barracks累计超过待办3，被原validator拒绝。这证明至少一条
重复/超额现象在原始权重也会出现，不支持笼统说“都是RL训练坏了”。
原始seed12007缺开始标签，但有EOS和结束标签；现有validator仍接受，其Marine
在先，未计入完整纠错。上述“validator接受”和“EOS/结束标签”不是严格标签成对指标。

**结论：A4在本输入这8个配对抽样上没有整体更差；稳定纠错仍未通过。**
不能由3/8→5/8宣布一般性RL收益；更不能把旧Level7 step24的另一输入退化证据
套用到本Level6 A4。该指标只检查本次首个人口前置，先补一个Depot不保证整段
大量Marine、Reactor及后续科技都能执行。本轮没有建成、战斗或胜率证据。

目前不需要据此回滚A4或采用上轮示例换序。下一项建议只读核对A4四步训练中
是否出现人口阻塞、不同前置顺序，以及对应reward/advantage是否区分；先确定
有没有提供稳定纠错的学习信号，不新增prompt/战术机制或GPU实验。

## 配对审计与归档

全部配置逐字段相同，8对输入token/CUDA RNG哈希相同；原始四rank推理前后
Adam状态空，旧A4四rank计数15的记录完整。重算8答validator/parser及配对指标
与result一致，旧A4登记文件哈希全部匹配。唯一生产差异为模型权重，正式模块
相对`0ac4c73`无diff。完整输出在43新RUN和本地
`/private/tmp/sc2-original-a4-results.OATxU0`，原始A4副本仍在
`/private/tmp/sc2-example-order-results.UvLUuA`；[机器记录和哈希](original-a4-r1-results.json)。
