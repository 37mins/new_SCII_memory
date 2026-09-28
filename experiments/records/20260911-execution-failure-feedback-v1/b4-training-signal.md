# B4：反馈进入了 RL，但纠错选择得到了什么信号

2026-09-12，只读产物/冻结源码和轻量 CPU 分析，无新游戏、GPU、PPO 或运行模块改动。

**B4 确实覆盖了一点人口阻塞并出现一次 Depot 优先选择，但覆盖少，且反馈后的
step 相对信号大多消失在单样本分组中。四批总体 advantage 非零不等于纠错回答
被有效区分。尚未发现新的实现 bug，也没有证据据此晋级长训。**

## 冻结对象

RUN `20260911-execution-failure-feedback-v1-spb2-original-l6-g2-B-rl4-4gpu-s0-r2`，
SHA `361cb7ae7fd86fe8811d6fcb6946d0241bf93e11`，43、GPU0–3、原始 SpB、Level6。
与 [A4](a4-training-signal.md) 的方法差异为失败反馈/配额释放/重新规划机制开启。
history=0/off、step shaping=off、GiGPO mean_norm、gamma=.95，其他冻结项见 [rl4.md](rl4.md)。
32 局只有四个游戏 seed，各重复八局，不当作 32 个独立场景。

43 的冻结部署 clean；manifest/config/原始回答/终局/种子/console/result 七文件哈希
与本地旧快照一致；`.done` 与 result 哈希相同，确认四步、退出0、step4保存完整。
未加载 checkpoint 张量。逐条证据、保守标量界和源码哈希见 [b4-training-signal.json](b4-training-signal.json)。
服务器 run：

```text
/root/new-scii-execution-feedback-local-20260911/shared/yangjinyue/new_SCII/experiments/20260911-execution-failure-feedback-v1/runs/20260911-execution-failure-feedback-v1-spb2-original-l6-g2-B-rl4-4gpu-s0-r2
```

## 反馈覆盖与真实回答

198 条真实回答，178 合法；74 条带失败反馈，70 条合法。
失败类别为 Reactor 缺气63、TechLab缺气1、Marauder缺TechLab6、Marine人口阻塞4。
人口余量0的 query 共5条：4条有阻塞反馈，另外1条是下一阶段普通调用。

| 阻塞反馈位置 | 模型选择 | 后续可确认的事实 |
| --- | --- | --- |
| step1/env5，06:34、07:14、07:53，日志行78/86/94 | 三次均 Marine2 → Depot → Barracks，均被validator接受 | 前两答之后再次收到相同人口阻塞、processed_prefix_count=0；整局仍 Victory、reward20.5 |
| step2/env6，07:02，日志行127 | Depot1 → Barracks1，不先安排耗人口单位 | 下一答07:03进入新阶段，人口仍0；整局 Victory、reward20.5。只能确认选择改变，不能据此说Depot已完成 |

行127原始回答：

```text
<answer>
- supplydepot x 1
- barracks x 1
</answer>
```

因此是 **1/4 条反馈后选择直接补给，3/4 未纠正顺序**，不是4次均不纠正。
这只是两个不同游戏/训练时点中的四次调用，不能当作同状态因果对照或学习成功率。
本轮最初口头计数漏掉了不含Marine的Depot-only回答，已纠正并加入专门CPU测试；
这是审计口径纠正，不是运行代码修复。

## 信号为什么不足

| global step / seed | 胜负 | 反馈回答 | 其中同输入只有自身 | 全批日志 advantage |
| --- | --- | --- | --- | --- |
| 1 / 2732 | 8/8 | 17 | 17 | −3.109～2.252 |
| 2 / 43567 | 8/8 | 11 | 11 | −2.320～1.865 |
| 3 / 42613 | 0/8 | 36 | 33 | −8.903～6.842 |
| 4 / 52416 | 7/8 | 10 | 10 | −64.868～20.238 |

`strategic_sc2_env_manager_v2.py:365` 传完整query为anchor；history关闭时日志prompt
与其一致。`rollout_loop.py:69,410` 保留anchor与每八局的group UID。
`gigpo/core_gigpo.py:247,338` 当前按组内**完全相同anchor**中心化，不做相似状态聚类。

**71/74条反馈的anchor只有一个真实样本，step项为“自身分数−自身分数”。**
随机补齐即便复制该条，也只是同分复制，不产生step排名；浮点残差不算有效排名。
这里不主张episode项、正则项、Adam动量或总参数更新都为零，也不把完整query分组
直接认定为bug，不能未经验证就合并不同状态。

唯一多样本反馈分组为 step3、05:53，env4/6/7，日志行181/183/184：

| env | 首段选择摘要 | 剩余决策数 | 整局reward | 当前答的未中心化折扣回报 |
| --- | --- | --- | --- | --- |
| 4 | Marine3、Depot1、Barracks1，不选Reactor | 6 | −49.450 | −38.2635 |
| 6 | Marine2、Depot2、Barracks1、Reactor2、SCV3、Refinery1 | 10 | −49.481 | −31.1854 |
| 7 | Marine38、Depot6、Barracks1，不选Reactor | 3 | −49.450 | −44.6286 |

env6 下一次又因 Reactor 缺气失败。它的整局分略差，但因为同类负终局回报被更多
Planner调用折扣，当前标量反而更高。**这不是已保存的完整advantage或参数梯度**；
补齐/平衡及episode项未重建。它说明非零step信号不天然等于“解决了失败”。
旧 [gamma=.95/1.0 A/B](../20260910-spb-fsdp-layer-wrap-v1/gamma-ab.md) 已完成且两组
均39/64胜，本轮不重复测试gamma或把它晋级为根因修复。

那条正确Depot选择（step2行127）具体得到什么：step项数学上为0；八局整局reward
均20.5、该局五答全合法。按冻结 `episode_norm_reward` 与无效惩罚实现，枚举可能的
episode代表行并保守考虑3条补齐，其episode项仅 **0～0.005**。上限来自其他轨迹的
代表行可能被扣0.1，不是额外确认Depot完成的奖励。step1的三条未纠正回答episode项
可能为−0.002083～0.009375；不能声称它们必然受到负强化。

这些是CPU标量界，不是原始张量：复制ID、平衡顺序、逐token advantage未保存。
当前奖励对胜局的解除阻塞/产兵没有直接独立记分；阶段/攻击/格式等分仍可能影响
累计值。此前 [Depot顺序实局对照](supply-order.md) 已证明实际产出差异却同为20.5，
这比跨step同分更强，但仍不能把奖励认作唯一原因。

## 结论与唯一下一项

反馈接口和可选动作并未全部失效：它们提供了改选机会，原模型也能生成正确前置
选择。更明确的瓶颈是**纠错状态覆盖少、反馈分支缺组内对照、现有回报不能稳定
识别有效纠错**。B4独评2/8、A4为5/8只是小样本结果，不能凭此确定优化器bug或
普遍退化；本轮没有恢复任何参数更新证据。

下一项沿用已有 [奖励审计路线](reward-discrimination.md)：**先形成并离线审查一个
基于真实解除阻塞的过程奖励候选，暂不实施或上卡**。明确什么是模型所选动作、
什么是实际执行进展、何时结算一次；用既有重复阻塞、无产出补给和规则补项日志
检验刷分反例。不能按出现Depot/Refinery关键词给分，不能复活逐call状态累加，
不能只因删除失败项或提交命令就宣称已解决。口径与公开依据核验后再提交单点A/B
及预算供用户确认，不同时改prompt、anchor分组、gamma、战术或采样。

CPU：`audit_b4_training_signal.py --snapshot <B-run小文件副本> --output <新JSON>`
核对198答/32终局/四批均值通过；`test_b4_training_signal.py` 四项通过。
