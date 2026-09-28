# 执行改进能否进入训练奖励

2026-09-11，只读离线结论；不改reward、loss或当前训练。
脚本 `audit_rl4_rewards.py` 校验r2的冻结SHA和policy归因文件哈希，读取已有小型日志；
按完成的global step和真实游戏边界汇总，终局奖励归到最后一个真实决策，
不制造样本。输出 [快照JSON](rl4-reward-audit-snapshot.json) 保存原始回答、反馈、
reward、来源行号及文件SHA256。快照覆盖A全部4步、B前2步；第三批尚未进入该快照，
不冒充B最终结果或精确PPO张量重放。
两组结束后已用同脚本生成[完整审计](rl4-reward-audit-final.json)：A/B各4步、
32局，A24胜/B23胜。旧快照保留，未覆盖。B共74条消费回答收到反馈，四步
advantage均非全零，但这仍不是纠错学习效果的证明。

| 已有证据 | 直接含义 | 不能推出 |
| --- | --- | --- |
| A step3：8局均−49.382；每局2条回答，完整bullet序列相同；记录advantage为0 | 这一批没有记录到可区分计划的相对优势 | 所有失败局都无学习空间 |
| A step4：8局均20.5；完整bullet序列3种；记录advantage为0 | 输出文字已有差异，但该批没有任务策略梯度信号 | 三种计划实际质量一定不同 |
| B step2：8局均20.5；每局3–8条回答；advantage −2.320至1.865 | 总回报相同不等于GiGPO联合advantage必为0 | 这些非零信号已学会有效纠错 |
| [Depot顺序对照](supply-order.md)：B均完成8Depot/新增2Marine，A完成0–1Depot/新增0Marine；两组均胜、reward均20.5 | 这个已确认的执行差异没有进入整局reward差异 | B的8Depot数量最优、所有seed均如此 |

## 冻结代码依据

runtime `361cb7ae7fd86fe8811d6fcb6946d0241bf93e11`，不是根据新工作树猜历史。

- `agents/planner.py:1077` 的 `game_end`：胜局固定+20，加最后格式合法的+0.5；
  仅非胜局增加 `final_state_shaping_reward`。胜局不直接奖励解除阻塞或单位产出。
- `agents/planner.py:1133`：本轮`SC2_STEP_SHAPING_MODE=off`，没有逐call状态加分。
- `agents/commander.py:282`：阶段速度、计划中的attack结算仍可能增加过程奖励。
  因而不能把所有胜局总reward都描述成固定20.5，更不能把这些分数直接等同战斗效果。
- `gigpo/core_gigpo.py:87,138,179,340`：episode相对回报加anchor-state相对折扣回报；
  gamma=.95按Planner transition计数。B step2每局均只有终局20.5，但轨迹长度不同，
  在共享anchor上可能得到不同折扣回报。这解释了“同总分仍有非零advantage”的来源，
  不是已证明的实现bug，也不是新的gamma候选；已有gamma1.0对照并未改善总胜率。

[GiGPO原论文（2025，§4.1–4.3）](https://arxiv.org/html/2505.10978v3#S4)
明确区分两级相对优势。这里仅核对其结构；实际代码的轨迹均分等本地适配不称为
与公开代码逐字一致。既有结果不支持仅因8局同分就换算法或混合不同seed强造优势。

## 当前判断和下一步

已经确认**部分真实执行改善没有对应奖励差**，但没有证明这是唯一根因。
反馈B前两批已有28条消费回答收到执行反馈；接口通了不代表权重新学到纠错。
A零优势批的grad_norm仍为0.021/0.025，不能说模型完全没更新：正则项和Adam动量
与任务策略梯度是不同来源。这里没有恢复参数梯度/Adam张量，不作数值根因结论。

先完成[原始/A4/B4同条件独立评测](learning-eval.md)。只有B不能改善纠错时，再提出
**一个**基于真实执行进展的过程奖励候选；先用日志检查重复刷建筑、重复阻塞再解除、
规则补兵冒充模型贡献等反例，再审批实局A/B。当前没有采用新reward公式，也不恢复
旧的逐call累计状态奖励。胜负持平而纠错提高时分别报告，不包装为整体学习优势。

本地实际命令：

```bash
python experiments/records/20260911-execution-failure-feedback-v1/audit_rl4_rewards.py \
  --snapshots /private/tmp/sc2-feedback-r2-audit.IKrMQp --output <未使用的JSON路径>
```

已运行，退出0；A32局24胜、B前16局16胜，均与已完成step日志均值一致。
对应ledger CPU回归通过；本分析没有使用GPU、checkpoint或额外SC2游戏。
完成后从`/private/tmp/sc2-feedback-r2-final.hrjkuX`补跑同命令，退出0，得到完整审计。
早期原始快照的私有备份位于43
`/root/new-scii-execution-feedback-20260911/rl4-reward-snapshot-20260911.tgz`，
SHA256 `738229b23b4d707ed3fddad461c50d9f48ef34f53148141e0211544513cde571`。

## 完整B中的反馈与具体选择（20:23补充）

[逐项统计及CPU探针](rl4-feedback-analysis.json)复用上述完成日志，不启新游戏。
B的74条反馈对应：Reactor缺气63、TechLab缺气1、Marauder缺TechLab6、Marine人口阻塞4。
70/74条后续回答被原validator判合法，但不等于实际可执行或解决前置条件。
各局内相邻反馈共47对，41对重复同一动作和同类原因，其中38对是Reactor缺气。
此分母是反馈对，不是“所有动作失败率”；中间可能有普通调用，不冒称立即下一命令。

缺气的64条query用冻结runtime原validator核验：`- refinery x 1`有59条可选且合法。
另5条自身已有1个Refinery、当前气36–48，不能据此认定系统错误地禁止补气。
实际16条原始回答包含Refinery，14条整体合法；其中11条把Reactor排在Refinery之前。
没有一条将其放第一，但并不预设所有场景都必须“Refinery第一”才正确。

具体例子：B step4/env0/seed52416，prompt_response第273行，05:36观测345矿、0气、
自身无Refinery；候选表明确`refinery x 1`。反馈为Reactor缺气0/50，模型回答：

```text
- marine x 8
- supplydepot x 7
- barracks x 2
- reactor barracks x 3
```

此回答与手写`- refinery x 1`在该完整query下都被原validator接受。
08:21再次反馈缺气，观测仍无自身Refinery，模型继续Marine/SCV/Reactor；该局最终败。
所以这个例子里不是完全没有可选动作，而是模型没有稳定优先解决实际缺口。
这不证明整局失败只由这一项造成，也不证明手写替代已经在真实SC2施工成功。

当前结论：反馈确实增加了参与和计划差异（A108回答、B198回答），B四步均有非零
advantage，但既没有训练胜率提升，也没有可靠纠错学习证据。两组同局数/global step，
不等于同生成token数或同优化器更新次数。应等独立原始/A4/B4评测判断权重是否学坏，
不从“非零advantage”直接晋级，也不凭本次分析改奖励或prompt。
